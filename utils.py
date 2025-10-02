import torch
import torch.nn.functional as F


def max_pool_attention_mask(attention_mask, stride=2):
    """
    Max-pool attention mask with the given stride to match encoder hidden state dimensions.

    Args:
        attention_mask (torch.Tensor): Tensor of shape (batch_size, seq_length) with 1s and 0s
        stride (int): Pooling stride, default 2

    Returns:
        torch.Tensor: Max-pooled attention mask of shape (batch_size, seq_length // stride)
    """
    if attention_mask is None:
        return None

    # Reshape to (batch_size, -1, stride) and apply max pooling
    batch_size, seq_length = attention_mask.shape

    # Reshape and apply max pooling
    pooled_mask = attention_mask.view(
        batch_size, -1, stride).any(dim=-1).float()

    return pooled_mask


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


def weighted_downsample(boundaries, hidden, weights, eps=1e-6):
    """Downsample hidden states using per-segment softmax weighting.

    Each segment is determined by ``boundaries`` in the same way as ``downsample``.
    Within a segment the provided ``weights`` are normalised with a softmax and
    used to take a weighted sum of the hidden states.

    Args:
        boundaries (torch.Tensor): Boundary tensor of shape (B, L) with ones
            marking the end of a segment at position *t*.
        hidden (torch.Tensor): Hidden state tensor of shape (B, L, D).
        weights (torch.Tensor): Logit weights for each timestep of shape (B, L)
            or (B, L, 1). Higher logits give more influence within a segment.
        eps (float): Numerical stability constant for divisions.

    Returns:
        torch.Tensor: Weighted segment summaries of shape (S, B, D) where
            ``S`` is the maximum number of segments in the batch.
    """
    batch_size, seq_len, model_dim = hidden.shape
    device = hidden.device
    dtype = hidden.dtype

    if seq_len == 0:
        return torch.zeros((0, batch_size, model_dim), device=device, dtype=dtype)

    boundaries = boundaries.long()

    if weights is None:
        raise ValueError("weights must be provided for weighted_downsample")

    if weights.dim() == 3 and weights.size(-1) == 1:
        weights = weights.squeeze(-1)

    if weights.shape != boundaries.shape:
        raise ValueError(
            f"weights must have shape {boundaries.shape}, but got {weights.shape}")

    weights = weights.to(device=device, dtype=dtype)

    # Segment IDs: segment 0 starts at index 0, new segment after each boundary.
    segment_ids = torch.cat([
        torch.zeros_like(boundaries[:, :1]),
        boundaries[:, :-1]
    ], dim=1).cumsum(dim=1)

    num_segments_per_batch = boundaries.sum(dim=1)
    max_segments = num_segments_per_batch.max().item() if batch_size > 0 else 0

    if max_segments == 0:
        return torch.zeros((0, batch_size, model_dim), device=device, dtype=dtype)

    max_segments = int(max_segments)

    valid_mask = segment_ids < num_segments_per_batch.unsqueeze(1)

    index_scatter = segment_ids.clamp(max=max_segments - 1)

    # Count valid elements per segment for masking and fallbacks.
    src_lengths = torch.ones_like(segment_ids) * valid_mask
    segment_lengths = torch.zeros(
        batch_size, max_segments, device=device, dtype=torch.int64)
    segment_lengths.scatter_add_(dim=1, index=index_scatter, src=src_lengths)

    finfo = torch.finfo(weights.dtype)
    masked_weights = torch.where(
        valid_mask, weights, torch.full_like(weights, finfo.min))

    segment_max = torch.full(
        (batch_size, max_segments), finfo.min, device=device, dtype=weights.dtype)

    if hasattr(segment_max, "scatter_reduce_"):
        segment_max.scatter_reduce_(
            dim=1,
            index=index_scatter,
            src=masked_weights,
            reduce="amax",
            include_self=True,
        )
    else:
        for b in range(batch_size):
            total_segments = int(num_segments_per_batch[b].item())
            for seg_idx in range(total_segments):
                seg_mask = (segment_ids[b] == seg_idx) & valid_mask[b]
                if seg_mask.any():
                    segment_max[b, seg_idx] = masked_weights[b][seg_mask].max()
                else:
                    segment_max[b, seg_idx] = 0.0

    # Replace max for empty segments so gather does not introduce -inf.
    segment_max = torch.where(
        segment_lengths > 0,
        segment_max,
        torch.zeros_like(segment_max)
    )

    gathered_segment_max = segment_max.gather(1, index_scatter)
    stable_logits = masked_weights - gathered_segment_max
    exp_weights = torch.exp(stable_logits)
    exp_weights = exp_weights * valid_mask

    segment_weight_sum = torch.zeros(
        batch_size, max_segments, device=device, dtype=weights.dtype)
    segment_weight_sum.scatter_add_(dim=1, index=index_scatter, src=exp_weights)

    safe_eps = max(eps, float(finfo.tiny))
    segment_weight_sum = torch.clamp(segment_weight_sum, min=safe_eps)

    weighted_hidden = hidden * exp_weights.unsqueeze(-1)

    summed_hidden = torch.zeros(
        batch_size, max_segments, model_dim, device=device, dtype=dtype)
    summed_hidden.scatter_add_(
        dim=1,
        index=index_scatter.unsqueeze(-1).expand_as(hidden),
        src=weighted_hidden
    )

    pooled_batch = summed_hidden / segment_weight_sum.unsqueeze(-1)

    return pooled_batch.transpose(0, 1)


# def downsample(boundaries, hidden, pooling="mean"):
#     """
#     Downsamples hidden states based on boundaries using vectorized torch operations.
#     A new segment starts at index 0 and at every index t+1 where boundaries[b, t] == 1.

#     Args:
#         boundaries (torch.Tensor): Tensor of shape (B, L), dtype=float or long.
#                                    1 indicates the *end* of a segment at index t.
#         hidden (torch.Tensor): Tensor of shape (B, L, D) containing hidden states.

#     Returns:
#         torch.Tensor: Pooled hidden states of shape (S, B, D), where S is the
#                       maximum number of segments in the batch. Padded with zeros.
#     """
#     batch_size, seq_len, model_dim = hidden.shape
#     device = hidden.device
#     dtype = hidden.dtype

#     # Ensure boundaries is long type for indexing and cumsum
#     boundaries = boundaries.long()

#     # 1. Calculate segment IDs for each time step t in each batch item b.
#     # segment_ids[b, t] = ID of the segment that hidden[b, t] belongs to.
#     # A new segment starts at t=0 and at t+1 if boundaries[b, t] == 1.
#     # We use cumsum on boundaries shifted right by one, prepended with 0.
#     segment_ids = torch.cat([
#         torch.zeros_like(boundaries[:, :1]),  # Segment 0 starts at index 0
#         # Boundary at t-1 means new segment starts at t
#         boundaries[:, :-1]
#     ], dim=1).cumsum(dim=1)  # Shape: (B, L)

#     # 2. Determine the number of segments for each batch item and the maximum.
#     num_segments_per_batch = segment_ids[:, -1] + 1  # Max segment ID + 1
#     max_segments = num_segments_per_batch.max(
#     ).item() if batch_size > 0 and seq_len > 0 else 0

#     if pooling == "mean":
#         # 3. Sum hidden states per segment using scatter_add_.
#         # Target tensor for sums: (B, max_segments, D)
#         summed_hidden = torch.zeros(
#             batch_size, max_segments, model_dim, device=device, dtype=dtype)
#         # Index for scatter_add_: needs to match shape of source (hidden) along scatter dim (1)
#         # Index shape: (B, L) -> expand to (B, L, D)
#         index = segment_ids.unsqueeze(-1).expand_as(hidden)  # Shape: (B, L, D)
#         summed_hidden.scatter_add_(dim=1, index=index, src=hidden)

#         # 4. Count segment lengths using scatter_add_.
#         # Target tensor for counts: (B, max_segments)
#         segment_lengths = torch.zeros(
#             batch_size, max_segments, device=device, dtype=torch.int64)
#         # Index for scatter_add_: (B, L)
#         # Source for scatter_add_: ones(B, L)
#         segment_lengths.scatter_add_(
#             dim=1, index=segment_ids, src=torch.ones_like(segment_ids))

#         # 5. Calculate the mean.
#         # Avoid division by zero for potentially empty segments (scatter_add initializes with 0).
#         # Clamp segment_lengths to minimum 1 before dividing.
#         # Shape: (B, max_segments, 1)
#         segment_lengths_clamped = segment_lengths.unsqueeze(-1).clamp(min=1)
#         # Shape: (B, max_segments, D)
#         pooled_batch = summed_hidden / segment_lengths_clamped
#     elif pooling == "max":
#         index = segment_ids.unsqueeze(-1).expand_as(hidden)  # Shape: (B, L, D)

#         pooled_batch = torch.scatter_reduce(
#             input=torch.full((batch_size, max_segments, model_dim), torch.finfo(
#                 dtype).min, device=device, dtype=dtype),  # Initialize with min value
#             dim=1,
#             index=index,
#             src=hidden,  # Use src argument for the source tensor
#             reduce="amax",
#             include_self=False  # Exclude initial min values from reduction
#         )

#     # 6. Transpose to match the expected output shape (S, B, D).
#     pooled_tensor = pooled_batch.transpose(0, 1)  # Shape: (max_segments, B, D)

#     return pooled_tensor


def delete(boundaries, hidden):
    """
    Keeps hidden states at boundary locations and deletes others.

    Args:
        boundaries (torch.Tensor): Tensor of shape (B, L), dtype=float or long.
                                   A value of 1 at index t indicates that hidden[b, t]
                                   is a boundary element and should be KEPT.
        hidden (torch.Tensor): Tensor of shape (B, L, D) containing hidden states.

    Returns:
        torch.Tensor: Hidden states with non-boundary elements removed, of shape (S, B, D),
                      where S is the maximum number of boundary elements
                      across all items in the batch. Padded with zeros.
    """
    batch_size, seq_len, model_dim = hidden.shape
    device = hidden.device
    dtype = hidden.dtype

    boundaries = boundaries.long()

    kept_batch_items = []

    for b in range(batch_size):
        b_hidden = hidden[b]  # Shape (L, D)
        b_boundaries = boundaries[b]  # Shape (L,)

        # Create a mask for elements to keep (where boundary is 1)
        # If boundaries[b, t] == 1, it's a boundary, so we keep it.
        keep_mask = (b_boundaries == 1)  # Shape (L,), boolean

        # Select elements to keep
        kept_elements = b_hidden[keep_mask]  # Shape (num_kept, D)
        kept_batch_items.append(kept_elements)

    # If all elements in all batch items were deleted, kept_batch_items would contain
    # tensors of shape (0, D). pad_sequence handles this by returning a tensor of shape (0, B, D).
    padded_kept_tensor = torch.nn.utils.rnn.pad_sequence(
        kept_batch_items, batch_first=False, padding_value=0.0
    )  # Shape (S, B, D)

    return padded_kept_tensor


def cross_attention_downsample(boundaries, hidden, cross_attn_query, cross_attn_key, cross_attn_value, cross_attn_scale):
    """
    Use cross-attention between boundary and non-boundary vectors for downsampling.
    Vectorized implementation without per-batch loops.

    Args:
        boundaries: [batch_size, seq_len] - binary mask indicating boundaries
        hidden: [batch_size, seq_len, hidden_dim] - hidden states
        cross_attn_query: Linear layer for computing queries
        cross_attn_key: Linear layer for computing keys
        cross_attn_value: Linear layer for computing values
        cross_attn_scale: Scaling factor for attention scores

    Returns:
        pooled: [batch_size, seq_len, hidden_dim] - downsampled representations
    """
    batch_size, seq_len, hidden_dim = hidden.shape
    device = hidden.device

    # Create masks for boundary and non-boundary positions
    boundary_mask = boundaries.bool()  # [batch_size, seq_len]
    non_boundary_mask = ~boundary_mask  # [batch_size, seq_len]

    # Compute queries, keys, values for all hidden states
    # [batch_size, seq_len, hidden_dim]
    all_queries = cross_attn_query(hidden)
    # [batch_size, seq_len, hidden_dim]
    all_keys = cross_attn_key(hidden)
    # [batch_size, seq_len, hidden_dim]
    all_values = cross_attn_value(hidden)

    # Create attention mask: boundary positions attend to non-boundary positions
    # [batch_size, seq_len, seq_len] - True where boundary can attend to non-boundary
    attn_mask = boundary_mask.unsqueeze(-1) & non_boundary_mask.unsqueeze(1)

    # Compute attention scores for all pairs
    # [batch_size, seq_len, seq_len]
    attn_scores = torch.bmm(
        all_queries, all_keys.transpose(-2, -1)) * cross_attn_scale

    # Apply mask: set invalid positions to very negative values
    attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))

    # For positions where no valid attention exists, we'll handle separately
    # Check if any boundary position has valid non-boundary positions to attend to
    has_valid_attn = attn_mask.any(dim=-1)  # [batch_size, seq_len]

    # Compute attention weights
    # [batch_size, seq_len, seq_len]
    attn_weights = F.softmax(attn_scores, dim=-1)
    # Set NaN values (from softmax of all -inf) to 0
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # Apply attention to get pooled representations
    # [batch_size, seq_len, hidden_dim]
    pooled_representations = torch.bmm(attn_weights, all_values)

    # For boundary positions: combine with original vectors (residual connection)
    # For non-boundary positions: keep original
    boundary_output = hidden + pooled_representations
    result = torch.where(
        boundary_mask.unsqueeze(-1) & has_valid_attn.unsqueeze(-1),
        boundary_output,
        hidden
    )

    # Only keep boundary positions in the final output
    # Create output tensor with only boundary positions
    pooled_list = []
    for b in range(batch_size):
        boundary_indices = torch.where(boundary_mask[b])[0]
        if len(boundary_indices) > 0:
            pooled_list.append(result[b][boundary_indices])
        else:
            # If no boundaries, return the full sequence
            pooled_list.append(result[b])

    # Find max length and pad
    if pooled_list:
        max_len = max(rep.shape[0] for rep in pooled_list)
        padded_representations = []
        for rep in pooled_list:
            if rep.shape[0] < max_len:
                padding = torch.zeros(max_len - rep.shape[0], hidden_dim,
                                      device=device, dtype=rep.dtype)
                rep = torch.cat([rep, padding], dim=0)
            padded_representations.append(rep)

        # Stack into a single tensor [batch_size, max_len, hidden_dim]
        pooled = torch.stack(padded_representations, dim=0)
    else:
        # Fallback case
        pooled = hidden

    return pooled
