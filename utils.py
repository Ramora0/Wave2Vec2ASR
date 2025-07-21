import torch


def downsample(boundaries, hidden, pooling="mean"):
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

    if pooling == "mean":
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
    elif pooling == "max":
        index = segment_ids.unsqueeze(-1).expand_as(hidden)  # Shape: (B, L, D)

        pooled_batch = torch.scatter_reduce(
            input=torch.full((batch_size, max_segments, model_dim), torch.finfo(
                dtype).min, device=device, dtype=dtype),  # Initialize with min value
            dim=1,
            index=index,
            src=hidden,  # Use src argument for the source tensor
            reduce="amax",
            include_self=False  # Exclude initial min values from reduction
        )

    # 6. Transpose to match the expected output shape (S, B, D).
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
