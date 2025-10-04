import torch


def downsample(boundaries, hidden):
    """
    Downsamples hidden states based on boundaries using vectorized torch operations.
    A new segment starts at index 0 and at every index t+1 where boundaries[b, t] == 1.

    This revised version handles trailing segments of zeros by ensuring they are not
    included in the output. The number of output segments for each batch item is
    equal to the number of 1s in its corresponding boundaries row.

    Args:
        boundaries (torch.Tensor): Tensor of shape (B, L), dtype=float or long.
                                   1 indicates the *end* of a segment at index t.
        hidden (torch.Tensor): Tensor of shape (B, L, D) containing hidden states.

    Returns:
        torch.Tensor: Pooled hidden states of shape (S, B, D), where S is the
                      maximum number of segments (sum of 1s) in the batch.
    """
    batch_size, seq_len, model_dim = hidden.shape
    device = hidden.device
    dtype = hidden.dtype

    # Handle empty sequence edge case
    if seq_len == 0:
        return torch.zeros((0, batch_size, model_dim), device=device, dtype=dtype)

    boundaries = boundaries.long()

    # 1. Calculate segment IDs for each time step t in each batch item b.
    # A new segment starts at t=0 and at t+1 if boundaries[b, t] == 1.
    segment_ids = torch.cat([
        torch.zeros_like(boundaries[:, :1]),  # Segment 0 starts at index 0
        boundaries[:, :-1]
    ], dim=1).cumsum(dim=1)  # Shape: (B, L)

    # 2. Determine the actual number of segments for each batch item.
    # This is the sum of 1s in each boundaries row.
    num_segments_per_batch = boundaries.sum(dim=1)  # Shape: (B,)
    max_segments = num_segments_per_batch.max().item() if batch_size > 0 else 0

    # Handle case where there are no segments at all (e.g., all-zero boundaries)
    if max_segments == 0:
        return torch.zeros((0, batch_size, model_dim), device=device, dtype=dtype)

    # Cast to int for tensor creation
    max_segments = int(max_segments)

    # 3. Create a mask to nullify contributions from trailing time steps.
    # A time step t is valid if its segment ID is less than the total number of
    # actual segments for that batch item.
    # Shape: (B, L)
    valid_mask = (segment_ids < num_segments_per_batch.unsqueeze(1))

    # 4. Sum hidden states per segment, using the mask to ignore trailing states.
    # Zero out the hidden states for invalid time steps before summing.
    src_hidden = hidden * valid_mask.unsqueeze(-1)

    # Target tensor for sums, sized to the actual max number of segments.
    summed_hidden = torch.zeros(
        batch_size, max_segments, model_dim, device=device, dtype=dtype)

    # Clamp segment_ids to avoid out-of-bounds errors in scatter_add_.
    # Contributions from invalid parts are 0 anyway due to the mask.
    index_scatter = segment_ids.clamp(max=max_segments - 1)
    summed_hidden.scatter_add_(
        dim=1, index=index_scatter.unsqueeze(-1).expand_as(src_hidden), src=src_hidden)

    # 5. Count segment lengths, using the mask to ignore trailing states.
    src_lengths = torch.ones_like(segment_ids) * valid_mask
    segment_lengths = torch.zeros(
        batch_size, max_segments, device=device, dtype=torch.int64)
    segment_lengths.scatter_add_(dim=1, index=index_scatter, src=src_lengths)

    # 6. Calculate the mean.
    # Avoid division by zero for any segment that might have a length of 0.
    segment_lengths_clamped = segment_lengths.unsqueeze(-1).clamp(min=1)
    pooled_batch = summed_hidden / segment_lengths_clamped  # Shape: (B, S, D)

    # 7. Transpose to match the expected output shape (S, B, D).
    return pooled_batch.transpose(0, 1)
