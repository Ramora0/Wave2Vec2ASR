import torch
from utils import downsample, common, final


LAST_DEBUG_INFO = None
BOUNDARY_THRESHOLD = 0.5


def _binarize_boundaries(boundaries):
    return (boundaries >= BOUNDARY_THRESHOLD).to(boundaries.dtype)

def _get_segment_info(boundaries_for_batch, L):
    """
    Finds boundary indices and calculates segment lengths for a single batch item.
    A boundary at index `i` means the segment ENDS at `i` (inclusive).
    """
    boundary_mask = boundaries_for_batch >= BOUNDARY_THRESHOLD
    boundary_indices = boundary_mask.nonzero(as_tuple=True)[0]
    if len(boundary_indices) == 0:
        return None, None

    segment_lengths = []
    last_boundary = -1
    for b_idx in boundary_indices:
        idx_val = b_idx.item()
        segment_lengths.append(idx_val - last_boundary)
        last_boundary = idx_val
    
    return boundary_indices, torch.tensor(segment_lengths, device=boundaries_for_batch.device)


def simulate_left_shift(boundaries, hidden, b, i, original_output):
    """
    Analytically calculates diffs for a left shift (i -> i-1).
    Boundary `i` ends Seg_k. Seg_{k+1} starts at i+1.
    Shift left means Seg_k now ends at i-1, losing h[i].
    Seg_{k+1} now starts at i, gaining h[i].
    """
    L = boundaries.shape[1]
    if not (i > 0 and boundaries[b, i-1] < BOUNDARY_THRESHOLD):
        return None

    boundary_indices, segment_lengths = _get_segment_info(boundaries[b], L)
    if boundary_indices is None: return None

    k = (boundary_indices == i).nonzero(as_tuple=True)[0]
    if len(k) == 0: return None
    k = k.item()

    if k >= len(segment_lengths) - 1: # Cannot shift the last boundary if it affects a non-existent next segment
        return None

    len_k = segment_lengths[k]
    len_k_plus_1 = segment_lengths[k+1]

    if len_k <= 1: return None # Cannot make a zero-length segment

    mean_k = original_output[k, b, :]
    mean_k_plus_1 = original_output[k+1, b, :]
    h_i = hidden[i, b, :]

    # Seg_k LOSES h[i]
    diff_k = (mean_k - h_i) / (len_k - 1)
    # Seg_{k+1} GAINS h[i]
    diff_k_plus_1 = (h_i - mean_k_plus_1) / (len_k_plus_1 + 1)

    return diff_k, diff_k_plus_1


def simulate_right_shift(boundaries, hidden, b, i, original_output):
    """
    Analytically calculates diffs for a right shift (i -> i+1).
    Boundary `i` ends Seg_k. Seg_{k+1} starts at i+1.
    Shift right means Seg_k now ends at i+1, gaining h[i+1].
    Seg_{k+1} now starts at i+2, losing h[i+1].
    """
    L = boundaries.shape[1]
    if not (i < L - 1 and boundaries[b, i+1] < BOUNDARY_THRESHOLD):
        return None

    boundary_indices, segment_lengths = _get_segment_info(boundaries[b], L)
    if boundary_indices is None: return None

    k = (boundary_indices == i).nonzero(as_tuple=True)[0]
    if len(k) == 0: return None
    k = k.item()

    if k >= len(segment_lengths) - 1: return None

    len_k = segment_lengths[k]
    len_k_plus_1 = segment_lengths[k+1]

    if len_k_plus_1 <= 1: return None

    mean_k = original_output[k, b, :]
    mean_k_plus_1 = original_output[k+1, b, :]
    h_i_plus_1 = hidden[i+1, b, :]

    # Seg_k GAINS h[i+1]
    diff_k = (h_i_plus_1 - mean_k) / (len_k + 1)
    # Seg_{k+1} LOSES h[i+1]
    diff_k_plus_1 = (mean_k_plus_1 - h_i_plus_1) / (len_k_plus_1 - 1)

    return diff_k, diff_k_plus_1


class FastFiniteDifferenceDownsample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, boundaries, hidden, gradient_scale, debug=False, debug_threshold=1e-5):
        output = downsample(boundaries, hidden)
        ctx.save_for_backward(boundaries, hidden)
        ctx.gradient_scale = gradient_scale
        ctx.debug = bool(debug)
        ctx.debug_threshold = float(debug_threshold)
        if ctx.debug:
            global LAST_DEBUG_INFO
            LAST_DEBUG_INFO = None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        boundaries, hidden = ctx.saved_tensors
        gradient_scale = ctx.gradient_scale
        B, L = boundaries.shape
        binary_boundaries = _binarize_boundaries(boundaries)
        grad_boundaries = torch.zeros_like(boundaries, dtype=torch.float32)
        original_output = downsample(binary_boundaries, hidden)
        debug_records = [] if ctx.debug else None
        debug_batches = [] if ctx.debug else None

        for b in range(B):
            boundary_mask_eq = (boundaries[b] == 1)
            boundary_indices = binary_boundaries[b].nonzero(as_tuple=True)[0]
            boundary_mask_thresh = (boundaries[b] >= BOUNDARY_THRESHOLD)
            if debug_batches is not None:
                debug_batches.append({
                    "batch": b,
                    "boundary_indices": boundary_indices.detach().cpu().tolist(),
                    "boundary_values": boundaries[b].detach().cpu().tolist(),
                    "boundary_indices_threshold": boundary_mask_thresh.nonzero(as_tuple=True)[0].detach().cpu().tolist(),
                    "boundary_indices_eq": boundary_mask_eq.nonzero(as_tuple=True)[0].detach().cpu().tolist(),
                })
            if len(boundary_indices) <= 1:
                continue

            # Don't process the last boundary of the whole sequence
            last_boundary_of_sequence = boundary_indices[-1].item()

            for i in boundary_indices:
                i = i.item()
                if i == last_boundary_of_sequence:
                    continue

                delta_left = 0.0
                delta_right = 0.0
                
                k = (boundary_indices == i).nonzero(as_tuple=True)[0].item()

                # Right shift
                diffs_right = simulate_right_shift(binary_boundaries, hidden, b, i, original_output)
                if diffs_right is not None:
                    diff_k, diff_k_plus_1 = diffs_right
                    d_right = (grad_output[k, b, :] * diff_k).sum()
                    d_right += (grad_output[k+1, b, :] * diff_k_plus_1).sum()
                    delta_right = d_right.item()

                # Left shift
                diffs_left = simulate_left_shift(binary_boundaries, hidden, b, i, original_output)
                if diffs_left is not None:
                    diff_k, diff_k_plus_1 = diffs_left
                    d_left = (grad_output[k, b, :] * diff_k).sum()
                    d_left += (grad_output[k+1, b, :] * diff_k_plus_1).sum()
                    delta_left = d_left.item()

                # Apply finite difference gradients
                if i > 0:
                    grad_boundaries[b, i-1] += delta_left * gradient_scale
                grad_boundaries[b, i] -= (delta_left + delta_right) * gradient_scale
                if i < L - 1:
                    grad_boundaries[b, i+1] += delta_right * gradient_scale

                if debug_records is not None:
                    debug_records.append({
                        "batch": b,
                        "boundary_index": i,
                        "segment_index": k,
                        "delta_left": delta_left,
                        "delta_right": delta_right,
                        "grad_after": float(grad_boundaries[b, i].item()),
                    })
        
        foo = common(binary_boundaries)
        if foo is None:
            grad_hidden = torch.zeros_like(hidden)
        else:
            bar = final(foo=foo).to(dtype=hidden.dtype)
            grad_hidden = torch.einsum('sbd,bls->lbd', grad_output, bar)

        if debug_records is not None:
            global LAST_DEBUG_INFO
            LAST_DEBUG_INFO = {
                "records": debug_records,
                "grad_boundaries": grad_boundaries.detach().cpu(),
                "grad_hidden": grad_hidden.detach().cpu(),
                "threshold": ctx.debug_threshold,
                "batch_boundaries": debug_batches,
            }

        return grad_boundaries, grad_hidden, None, None, None


def get_last_fast_downsample_debug():
    return LAST_DEBUG_INFO


def downsample_with_fast_finite_diff_grad(boundaries, hidden, gradient_scale=1.0, debug=False, debug_threshold=1e-5):
    return FastFiniteDifferenceDownsample.apply(boundaries, hidden, gradient_scale, debug, debug_threshold)
