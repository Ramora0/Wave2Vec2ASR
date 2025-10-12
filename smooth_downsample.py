import torch
import torch.nn.functional as F
from utils import downsample


class SmoothedDownsample(torch.autograd.Function):
    """
    Calls utils.downsample in forward pass.
    In backward pass, computes normal gradients for boundaries but smoothed gradients for hidden.
    """
    @staticmethod
    def forward(ctx, boundaries, hidden, kernel):
        # Call the original downsample function
        output = downsample(boundaries, hidden)

        # Save for backward
        ctx.save_for_backward(boundaries, hidden, kernel)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        boundaries, hidden, kernel = ctx.saved_tensors

        # Recompute forward with requires_grad to get natural gradients
        with torch.enable_grad():
            boundaries_copy = boundaries.detach().requires_grad_(True)
            hidden_copy = hidden.detach().requires_grad_(True)
            output = downsample(boundaries_copy, hidden_copy)

            # Compute gradients
            output.backward(grad_output)
            grad_boundaries = boundaries_copy.grad
            grad_hidden = hidden_copy.grad

        # Smooth the hidden gradient
        if grad_hidden is not None:
            T, B, C = grad_hidden.shape
            grad_permuted = grad_hidden.permute(1, 2, 0)  # (T, B, C) -> (B, C, T)
            k = kernel.shape[2]
            grad_hidden_smoothed = F.conv1d(
                grad_permuted,
                weight=kernel,
                padding=k // 2,
                groups=C
            )
            grad_hidden_smoothed = grad_hidden_smoothed.permute(2, 0, 1)  # (B, C, T) -> (T, B, C)
        else:
            grad_hidden_smoothed = None

        return grad_boundaries, grad_hidden_smoothed, None


def downsample_with_smoothed_grad(boundaries, hidden, smoothing_kernel_size=3):
    """
    Downsamples a sequence using utils.downsample().

    Boundaries get normal PyTorch autograd (differentiable).
    Hidden states get smoothed gradients via custom backward pass.

    Args:
        boundaries (torch.Tensor): Boundary tensor of shape (B, T),
                                   where 1 indicates the start of a new segment.
        hidden (torch.Tensor): Input tensor of shape (T, B, C).
        smoothing_kernel_size (int): The size of the kernel for smoothing the gradient.

    Returns:
        torch.Tensor: The downsampled tensor of shape (S, B, C).
    """
    if smoothing_kernel_size % 2 == 0:
        raise ValueError("smoothing_kernel_size must be an odd number.")

    T, B, C = hidden.shape

    # Create smoothing kernel
    avg_kernel = torch.ones(C, 1, smoothing_kernel_size,
                            device=hidden.device, dtype=hidden.dtype) / smoothing_kernel_size

    # Apply custom downsample with smoothed gradients
    return SmoothedDownsample.apply(boundaries, hidden, avg_kernel)
