import torch
from utils import downsample, common, final


def _shift_boundary(boundaries, batch_idx, from_idx, to_idx):
    """
    Create a copy of boundaries with a boundary moved from one position to another.

    Args:
        boundaries: (B, L) tensor of boundary indicators
        batch_idx: Which batch item to modify
        from_idx: Current position of boundary (should have value 1)
        to_idx: New position for boundary (should have value 0)

    Returns:
        Modified boundaries tensor with boundary shifted
    """
    shifted = boundaries.clone()
    shifted[batch_idx, from_idx] = 0
    shifted[batch_idx, to_idx] = 1
    return shifted


def compute_pooled_state_diff_for_left_shift(boundaries, hidden, original_output, b, i, seg_idx):
    """
    Compute the difference in the two affected pooled states when boundary i shifts left.

    When a boundary at position i moves to i-1:
    - The pooled state before the boundary (seg_idx-1) LOSES the frame at position i-1
    - The pooled state after the boundary (seg_idx) GAINS the frame at position i-1

    Args:
        boundaries: (B, L) tensor of boundary indicators
        hidden: (L, B, D) tensor of hidden states
        original_output: (S, B, D) tensor of original downsampled output
        b: Batch index
        i: Position of boundary to shift
        seg_idx: Index of the segment starting at this boundary

    Returns:
        Tuple of (diff_before, diff_after) where each is a (D,) tensor,
        or None if shift is invalid
    """
    if i > 0 and boundaries[b, i-1] == 0:
        shifted_left = _shift_boundary(boundaries, b, i, i-1)
        output_after_shift = downsample(shifted_left, hidden)
        diff_before = output_after_shift[seg_idx-1, b, :] - original_output[seg_idx-1, b, :]
        diff_after = output_after_shift[seg_idx, b, :] - original_output[seg_idx, b, :]
        return (diff_before, diff_after)
    return None


def compute_pooled_state_diff_for_right_shift(boundaries, hidden, original_output, b, i, seg_idx):
    """
    Compute the difference in the two affected pooled states when boundary i shifts right.

    When a boundary at position i moves to i+1:
    - The pooled state before the boundary (seg_idx-1) GAINS the frame at position i
    - The pooled state after the boundary (seg_idx) LOSES the frame at position i

    Args:
        boundaries: (B, L) tensor of boundary indicators
        hidden: (L, B, D) tensor of hidden states
        original_output: (S, B, D) tensor of original downsampled output
        b: Batch index
        i: Position of boundary to shift
        seg_idx: Index of the segment starting at this boundary

    Returns:
        Tuple of (diff_before, diff_after) where each is a (D,) tensor,
        or None if shift is invalid
    """
    L = boundaries.shape[1]
    if i < L - 1 and boundaries[b, i+1] == 0:
        shifted_right = _shift_boundary(boundaries, b, i, i+1)
        output_after_shift = downsample(shifted_right, hidden)
        diff_before = output_after_shift[seg_idx-1, b, :] - original_output[seg_idx-1, b, :]
        diff_after = output_after_shift[seg_idx, b, :] - original_output[seg_idx, b, :]
        return (diff_before, diff_after)
    return None


class FiniteDifferenceDownsample(torch.autograd.Function):
    """
    Custom autograd function for downsampling with finite difference gradients.

    Forward pass: Uses utils.downsample for mean pooling based on boundaries
    Backward pass: Estimates boundary gradients using finite differences by trying
                   to shift each boundary left/right and measuring loss change
    """

    @staticmethod
    def forward(ctx, boundaries, hidden, gradient_scale, debug=False, debug_threshold=1e-5):
        """
        Forward pass using standard downsample operation.

        Args:
            boundaries: (B, L) tensor where 1 indicates segment start
            hidden: (L, B, D) tensor of hidden states
            gradient_scale: Scalar to scale the finite difference gradients

        Returns:
            output: (S, B, D) downsampled tensor
        """
        # Call the original downsample function from utils.py
        output = downsample(boundaries, hidden)

        # Save for backward
        ctx.save_for_backward(boundaries, hidden)
        ctx.gradient_scale = gradient_scale
        ctx.debug = bool(debug)
        ctx.debug_threshold = float(debug_threshold)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using finite difference approximation for boundary gradients.

        For each boundary at position i, we:
        1. Try shifting it left (i-1) and compute output change
        2. Try shifting it right (i+1) and compute output change
        3. Estimate loss change via dot product with grad_output
        4. Apply gradients: grad[i-1] += delta_left
                           grad[i] -= (delta_left + delta_right)
                           grad[i+1] += delta_right
        """
        boundaries, hidden = ctx.saved_tensors
        gradient_scale = ctx.gradient_scale

        B, L = boundaries.shape

        # Initialize gradient tensor for boundaries
        grad_boundaries = torch.zeros_like(boundaries, dtype=torch.float32)

        # Compute original output for comparison
        original_output = downsample(boundaries, hidden)

        # Iterate through each batch and each position
        for b in range(B):
            # Find the last boundary position for this batch
            boundary_positions = (boundaries[b] == 1).nonzero(as_tuple=True)[0]
            last_boundary_pos = boundary_positions[-1].item() if len(boundary_positions) > 0 else -1

            seg_idx = 1  # Track which segment starts after this boundary (first boundary ends segment 0, starts segment 1)
            for i in range(L):
                # Only process positions that have a boundary
                if boundaries[b, i] != 1:
                    continue

                # Skip the last boundary (it's static)
                if i == last_boundary_pos:
                    break

                delta_left = 0.0
                delta_right = 0.0

                # Try shifting boundary LEFT (i -> i-1)
                # Returns (diff_before, diff_after) for the two affected pooled states
                diffs_left = compute_pooled_state_diff_for_left_shift(
                    boundaries, hidden, original_output, b, i, seg_idx
                )
                if diffs_left is not None:
                    diff_before, diff_after = diffs_left
                    # Calculate gradient contribution from both affected segments
                    # seg_idx-1 is the segment ending at this boundary, seg_idx is the one after
                    delta_left = (grad_output[seg_idx - 1, b, :] * diff_before).sum().item()
                    delta_left += (grad_output[seg_idx, b, :] * diff_after).sum().item()

                # Try shifting boundary RIGHT (i -> i+1)
                # Returns (diff_before, diff_after) for the two affected pooled states
                diffs_right = compute_pooled_state_diff_for_right_shift(
                    boundaries, hidden, original_output, b, i, seg_idx
                )
                if diffs_right is not None:
                    diff_before, diff_after = diffs_right
                    # Calculate gradient contribution from both affected segments
                    # seg_idx-1 is the segment ending at this boundary, seg_idx is the one after
                    delta_right = (grad_output[seg_idx - 1, b, :] * diff_before).sum().item()
                    delta_right += (grad_output[seg_idx, b, :] * diff_after).sum().item()

                seg_idx += 1

                # Apply finite difference gradients
                if i > 0:
                    grad_boundaries[b, i-1] += delta_left * gradient_scale

                grad_boundaries[b, i] -= (delta_left +
                                          delta_right) * gradient_scale

                if i < L - 1:
                    grad_boundaries[b, i+1] += delta_right * gradient_scale

        # Compute gradients w.r.t. hidden states via the downsample weights
        foo = common(boundaries)
        if foo is None:
            grad_hidden = torch.zeros_like(hidden)
        else:
            bar = final(foo=foo).to(dtype=hidden.dtype)
            grad_hidden = torch.einsum('sbd,bls->lbd', grad_output, bar)

        # Return gradients for (boundaries, hidden, gradient_scale)
        return grad_boundaries, grad_hidden, None, None, None


def downsample_with_finite_diff_grad(boundaries, hidden, gradient_scale=1.0, debug=False, debug_threshold=1e-5):
    """
    Downsample a sequence using mean pooling with finite difference gradients for boundaries.

    Forward pass: Uses utils.downsample() for standard mean pooling based on boundaries.
    Backward pass: Computes boundary gradients using finite differences by trying to
                   shift each boundary left/right and estimating the loss change.

    Args:
        boundaries (torch.Tensor): Boundary tensor of shape (B, L),
                                   where 1 indicates the start of a new segment.
        hidden (torch.Tensor): Input tensor of shape (L, B, D).
        gradient_scale (float): Scaling factor for the finite difference gradients.
                                Default is 1.0.

    Returns:
        torch.Tensor: The downsampled tensor of shape (S, B, D).
    """
    return FiniteDifferenceDownsample.apply(boundaries, hidden, gradient_scale, debug, debug_threshold)
