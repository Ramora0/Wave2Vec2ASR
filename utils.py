from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F


def _normalize_text(text: str) -> str:
    return text.strip().lower()


def tokenize_batch_texts(texts: Iterable[str], tokenizer) -> Any:
    normalized = [_normalize_text(text) for text in texts]
    return tokenizer(normalized, padding="longest", truncation=True)


def recover_text_from_feature(
    feature: Dict[str, Any],
    labels_row: torch.Tensor,
    attention_row: torch.Tensor,
    tokenizer,
    decoder_start_token_id: int,
) -> str:
    text = feature.get("text") if isinstance(feature, dict) else None

    if isinstance(text, (list, tuple)):
        text = " ".join(map(str, text))
    elif text is not None:
        text = str(text)

    if text is not None and text.strip():
        return text.strip()

    mask = attention_row == 1
    valid_ids = labels_row[mask].tolist()

    if valid_ids and valid_ids[0] == decoder_start_token_id:
        valid_ids = valid_ids[1:]

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        valid_ids = [tok for tok in valid_ids if tok != pad_id]

    decoded = tokenizer.decode(valid_ids, skip_special_tokens=True)
    return decoded.strip()


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


def final(foo,
          upsample):
    """
        Input:
            B x L x S
    """
    autoregressive = foo != 0
    lel = 1 - foo

    lel[autoregressive] = 0

    dim = 2 if upsample else 1

    lel = lel / (lel.sum(dim=dim, keepdim=True) + 1e-9)

    return lel


def common(boundaries, upsample=False):
    boundaries = boundaries.clone()

    n_segments = boundaries.sum(dim=-1).max().item()

    if upsample:
        n_segments += 1

    if n_segments == 0:
        return None

    tmp = torch.zeros_like(
        boundaries
    ).unsqueeze(2) + torch.arange(
        start=0,
        end=n_segments,
        device=boundaries.device
    )

    hh1 = boundaries.cumsum(1)

    if not upsample:
        hh1 -= boundaries

    foo = tmp - hh1.unsqueeze(-1)

    return foo


def downsample(boundaries, hidden):
    """
        Downsampling

        - The first element of boundaries tensor is always 0 and doesn't matter
        - 1 starts a new group

        Input:
            boundaries: B x L
            hidden: L x B x D
        Output:
            shortened_hidden: S x B x D
    """

    foo = common(boundaries, upsample=False)  # B x L x S

    if foo is None:
        return hidden.new_zeros(0, hidden.size(1), hidden.size(2))
    else:
        bar = final(foo=foo, upsample=False)  # B x L x S

        shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, bar)

        return shortened_hidden


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
