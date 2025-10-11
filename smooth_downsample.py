import torch
import torch.nn.functional as F


def downsample_with_smoothed_grad(boundaries, hidden, smoothing_kernel_size=3):
    """
    Downsamples a sequence using the exact (if idiosyncratic) forward pass logic
    from the reference `downsample` function, including discarding the last segment.

    The backward pass provides a custom smoothed gradient.

    Args:
        boundaries (torch.Tensor): Boundary tensor of shape (B, T),
                                   where 1 indicates the start of a new segment.
        hidden (torch.Tensor): Input tensor of shape (T, B, C).
        smoothing_kernel_size (int): The size of the kernel for smoothing the gradient.

    Returns:
        torch.Tensor: The downsampled tensor of shape (S, B, C).
    """

    class SmoothedDownsampleSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, boundaries, hidden, kernel):
            # --- Start: Reference Implementation Logic ---
            T, B, C = hidden.shape
            input_dtype = hidden.dtype

            # 1. `common` function logic
            n_segments = boundaries.sum(dim=-1).max().item()
            if n_segments == 0:
                return torch.empty(0, B, C, device=hidden.device, dtype=input_dtype)

            tmp = torch.zeros_like(boundaries).unsqueeze(2) + torch.arange(
                start=0, end=n_segments, device=boundaries.device
            )
            hh1 = boundaries.cumsum(1)
            hh1 -= boundaries  # Exclusive cumsum
            foo = tmp - hh1.unsqueeze(-1)

            # 2. `final` function logic
            bar = (foo == 0).to(dtype=input_dtype)
            bar = bar / (bar.sum(dim=1, keepdim=True) + 1e-9)

            # 3. `einsum` calculation
            # Permute hidden from (T, B, C) -> (T, B, C) to match einsum
            # Note: reference used 'lbd,bls->sbd', we use 'tbc,bts->sbc'
            shortened_hidden = torch.einsum('tbc,bts->sbc', hidden, bar)

            # --- End: Reference Implementation Logic ---

            # Save context for backward pass
            ctx.save_for_backward(bar, kernel)
            ctx.hidden_shape = hidden.shape

            return shortened_hidden

        @staticmethod
        def backward(ctx, grad_output):
            # grad_output has shape (S-1, B, C)
            bar, kernel = ctx.saved_tensors
            T, B, C = ctx.hidden_shape

            grad_input = torch.einsum('sbc,bts->tbc', grad_output, bar)

            # Apply smoothing
            # Permute for conv1d: (T, B, C) -> (B, C, T)
            grad_input_permuted = grad_input.permute(1, 2, 0)

            k = kernel.shape[2]
            grad_input_smoothed = F.conv1d(
                grad_input_permuted,
                weight=kernel,
                padding=k // 2,
                groups=C
            )

            # Permute back: (B, C, T) -> (T, B, C)
            grad_input_smoothed = grad_input_smoothed.permute(2, 0, 1)

            # Gradients for (boundaries, hidden, kernel)
            return None, grad_input_smoothed, None

    # --- Main function logic ---
    if smoothing_kernel_size % 2 == 0:
        raise ValueError("smoothing_kernel_size must be an odd number.")

    _T, _B, C = hidden.shape

    # Create a simple, non-learnable averaging kernel for smoothing the gradient
    avg_kernel = torch.ones(C, 1, smoothing_kernel_size,
                            device=hidden.device, dtype=hidden.dtype) / smoothing_kernel_size

    # Apply the custom downsampling function
    return SmoothedDownsampleSTE.apply(boundaries, hidden, avg_kernel)
