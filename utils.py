import torch
import torch.nn.functional as F


def downsample(boundaries, hidden):
    """
    Downsamples hidden states based on boundaries using vectorized torch operations.
    A new segment starts at index 0 and at every index t+1 where boundaries[b, t] == 1.

    Args:
        boundaries (torch.Tensor): Tensor of shape (B, L), dtype=float or long.
                                   1 indicates the *end* of a segment at index t.
        hidden (torch.Tensor): Tensor of shape (B, L, D) containing hidden states.

    Returns:
        torch.Tensor: Pooled hidden states of shape (S, B, D), where S is the
                      maximum number of segments in the batch. Padded with zeros.
    """
    batch_size, seq_len, model_dim = hidden.shape
    device = hidden.device
    dtype = hidden.dtype

    # Ensure boundaries is long type for indexing and cumsum
    boundaries = boundaries.long()

    # 1. Calculate segment IDs for each time step t in each batch item b.
    # segment_ids[b, t] = ID of the segment that hidden[b, t] belongs to.
    # A new segment starts at t=0 and at t+1 if boundaries[b, t] == 1.
    # We use cumsum on boundaries shifted right by one, prepended with 0.
    segment_ids = torch.cat([
        torch.zeros_like(boundaries[:, :1]),  # Segment 0 starts at index 0
        # Boundary at t-1 means new segment starts at t
        boundaries[:, :-1]
    ], dim=1).cumsum(dim=1)  # Shape: (B, L)

    # 2. Determine the number of segments for each batch item and the maximum.
    num_segments_per_batch = segment_ids[:, -1] + 1  # Max segment ID + 1
    max_segments = num_segments_per_batch.max(
    ).item() if batch_size > 0 and seq_len > 0 else 0

    # 3. Sum hidden states per segment using scatter_add_.
    # Target tensor for sums: (B, max_segments, D)
    summed_hidden = torch.zeros(
        batch_size, max_segments, model_dim, device=device, dtype=dtype)
    # Index for scatter_add_: needs to match shape of source (hidden) along scatter dim (1)
    # Index shape: (B, L) -> expand to (B, L, D)
    index = segment_ids.unsqueeze(-1).expand_as(hidden)  # Shape: (B, L, D)
    summed_hidden.scatter_add_(dim=1, index=index, src=hidden)

    # 4. Count segment lengths using scatter_add_.
    # Target tensor for counts: (B, max_segments)
    segment_lengths = torch.zeros(
        batch_size, max_segments, device=device, dtype=torch.int64)
    # Index for scatter_add_: (B, L)
    # Source for scatter_add_: ones(B, L)
    segment_lengths.scatter_add_(
        dim=1, index=segment_ids, src=torch.ones_like(segment_ids))

    # 5. Calculate the mean.
    # Avoid division by zero for potentially empty segments (scatter_add initializes with 0).
    # Clamp segment_lengths to minimum 1 before dividing.
    # Shape: (B, max_segments, 1)
    segment_lengths_clamped = segment_lengths.unsqueeze(-1).clamp(min=1)
    # Shape: (B, max_segments, D)
    pooled_batch = summed_hidden / segment_lengths_clamped

    # 6. Transpose to match the expected output shape (S, B, D).
    pooled_tensor = pooled_batch.transpose(0, 1)  # Shape: (max_segments, B, D)

    return pooled_tensor


def weighted_downsample(boundaries, probs, hidden):
    """
    Downsamples hidden states based on boundaries using weighted sums based on probability values.
    A new segment starts at index 0 and at every index t+1 where boundaries[b, t] == 1.

    Args:
        boundaries (torch.Tensor): Tensor of shape (B, L), dtype=float or long.
                                   1 indicates the *end* of a segment at index t.
        probs (torch.Tensor): Tensor of shape (B, L), dtype=float.
                              Probability values used for weighting the hidden states.
        hidden (torch.Tensor): Tensor of shape (B, L, D) containing hidden states.

    Returns:
        torch.Tensor: Weighted pooled hidden states of shape (S, B, D), where S is the
                      maximum number of segments in the batch. Padded with zeros.
    """
    batch_size, seq_len, model_dim = hidden.shape
    device = hidden.device
    dtype = hidden.dtype

    # Ensure boundaries is long type for indexing and cumsum
    boundaries = boundaries.long()

    # 1. Calculate segment IDs for each time step t in each batch item b.
    # segment_ids[b, t] = ID of the segment that hidden[b, t] belongs to.
    # A new segment starts at t=0 and at t+1 if boundaries[b, t] == 1.
    # We use cumsum on boundaries shifted right by one, prepended with 0.
    segment_ids = torch.cat([
        torch.zeros_like(boundaries[:, :1]),  # Segment 0 starts at index 0
        # Boundary at t-1 means new segment starts at t
        boundaries[:, :-1]
    ], dim=1).cumsum(dim=1)  # Shape: (B, L)

    # 2. Determine the number of segments for each batch item and the maximum.
    num_segments_per_batch = segment_ids[:, -1] + 1  # Max segment ID + 1
    max_segments = num_segments_per_batch.max(
    ).item() if batch_size > 0 and seq_len > 0 else 0

    # 3. Apply vectorized softmax within each segment
    # Create a unique segment identifier across the entire batch
    # Shape: (B, L) -> (B*L,)
    batch_indices = torch.arange(
        batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
    global_segment_ids = batch_indices * \
        max_segments + segment_ids  # Shape: (B, L)

    # Flatten for scatter operations
    flat_probs = probs.view(-1)  # Shape: (B*L,)
    flat_segment_ids = global_segment_ids.view(-1)  # Shape: (B*L,)

    # Apply softmax within each segment using scatter operations
    # First, find the maximum value within each segment for numerical stability
    max_vals = torch.full((batch_size * max_segments,),
                          float('-inf'), device=device, dtype=flat_probs.dtype)
    max_vals.scatter_reduce_(
        0, flat_segment_ids, flat_probs, reduce='amax', include_self=False)
    segment_maxes = max_vals[flat_segment_ids]  # Shape: (B*L,)

    # Subtract max for numerical stability and compute exp
    stable_probs = torch.exp(flat_probs - segment_maxes)  # Shape: (B*L,)

    # Sum exp values within each segment
    exp_sums = torch.zeros(batch_size * max_segments,
                           device=device, dtype=stable_probs.dtype)
    exp_sums.scatter_add_(0, flat_segment_ids, stable_probs)
    segment_sums = exp_sums[flat_segment_ids]  # Shape: (B*L,)

    # Compute normalized probabilities (softmax)
    normalized_flat = stable_probs / \
        segment_sums.clamp(min=1e-8)  # Shape: (B*L,)
    normalized_probs = normalized_flat.view(
        batch_size, seq_len)  # Shape: (B, L)

    # 4. Weight the hidden states by normalized probabilities before summing
    # Shape: (B, L, D)
    weighted_hidden = hidden * normalized_probs.unsqueeze(-1)

    # 5. Sum weighted hidden states per segment using scatter_add_.
    # Target tensor for weighted sums: (B, max_segments, D)
    weighted_summed_hidden = torch.zeros(
        batch_size, max_segments, model_dim, device=device, dtype=dtype)
    # Index for scatter_add_: needs to match shape of source (weighted_hidden) along scatter dim (1)
    # Index shape: (B, L) -> expand to (B, L, D)
    # Shape: (B, L, D)
    index = segment_ids.unsqueeze(-1).expand_as(weighted_hidden)
    weighted_summed_hidden.scatter_add_(
        dim=1, index=index, src=weighted_hidden)

    # 6. Since probabilities are now normalized within each segment (sum to 1),
    # the weighted sum IS the weighted average - no need to divide by weight sums
    # Shape: (B, max_segments, D)
    pooled_batch = weighted_summed_hidden

    # 7. Transpose to match the expected output shape (S, B, D).
    pooled_tensor = pooled_batch.transpose(0, 1)  # Shape: (max_segments, B, D)

    return pooled_tensor


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
