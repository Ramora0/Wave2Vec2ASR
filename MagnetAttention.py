import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.cache_utils import EncoderDecoderCache


class MagnetAttention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        query_mask_1d: Optional[torch.Tensor] = None,
        key_mask_1d: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel

        Args:
            query_mask_1d: 1D mask of shape (batch, query_seq_len) where 1 = valid, 0 = padded.
                          Prevents padded query positions from attending to ANY tokens.
            key_mask_1d: 1D mask of shape (batch, key_seq_len) where 1 = valid, 0 = padded.
                        Prevents ANY query from attending to padded key positions.

        For self-attention: typically only query_mask_1d is provided (queries and keys are same sequence)
        For cross-attention: both masks should be provided (decoder queries, encoder keys)
        """

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(
            bsz, tgt_len, self.num_heads, self.head_dim)
        query_states = query_states.transpose(1, 2).contiguous()

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states, value_states = past_key_value[self.layer_idx]
        else:
            key_states = self.k_proj(current_states).view(
                bsz, -1, self.num_heads, self.head_dim)
            value_states = self.v_proj(current_states).view(
                bsz, -1, self.num_heads, self.head_dim)
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {
                        "cache_position": cache_position}
                )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        # Apply padding masks using 1D masks without creating full 2D mask
        # attn_weights shape: (batch, num_heads, tgt_len, src_len)

        # Mask rows where queries are padded (prevents padded queries from attending)
        if query_mask_1d is not None:
            # query_mask_1d shape: (batch, tgt_len) where 1 = valid, 0 = padded
            # Shape: (batch, tgt_len) -> (batch, 1, tgt_len, 1) for broadcasting
            padded_queries = (query_mask_1d == 0).unsqueeze(1).unsqueeze(-1)
            attn_weights = attn_weights.masked_fill(padded_queries, float('-inf'))

        # Mask columns where keys are padded (prevents attending to padded keys)
        if key_mask_1d is not None:
            # key_mask_1d shape: (batch, src_len) where 1 = valid, 0 = padded
            # Shape: (batch, src_len) -> (batch, 1, 1, src_len) for broadcasting
            padded_keys = (key_mask_1d == 0).unsqueeze(1).unsqueeze(1)
            attn_weights = attn_weights.masked_fill(padded_keys, float('-inf'))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Replace NaN values with 0 (occurs when entire row is -inf for padded positions)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Verify that padded positions have zero attention weights
        # Check query mask (padded queries shouldn't attend to anything)
        if query_mask_1d is not None:
            # query_mask_1d shape: (batch, tgt_len)
            # attn_weights shape: (batch, num_heads, tgt_len, src_len)
            padded_queries_mask = (query_mask_1d == 0)  # (batch, tgt_len)

            if padded_queries_mask.any():
                # Check attention weights FROM padded query positions (rows should be zero)
                # Shape: (batch, tgt_len) -> (batch, 1, tgt_len, 1) for broadcasting
                padded_queries = padded_queries_mask.unsqueeze(1).unsqueeze(-1)
                attn_from_padded = attn_weights * padded_queries
                max_attn_from_padded = attn_from_padded.abs().max().item()

                assert max_attn_from_padded < 1e-5, (
                    f"Padded query positions are attending to others! Max attention weight: {max_attn_from_padded:.6e}. "
                    f"Attention weights FROM padded query positions should all be zero."
                )

        # Check key mask (nothing should attend to padded keys)
        if key_mask_1d is not None:
            # key_mask_1d shape: (batch, src_len)
            # attn_weights shape: (batch, num_heads, tgt_len, src_len)
            padded_keys_mask = (key_mask_1d == 0)  # (batch, src_len)

            if padded_keys_mask.any():
                # Check attention weights TO padded key positions (columns should be zero)
                # Shape: (batch, src_len) -> (batch, 1, 1, src_len) for broadcasting
                padded_keys = padded_keys_mask.unsqueeze(1).unsqueeze(1)
                attn_to_padded = attn_weights * padded_keys
                max_attn_to_padded = attn_to_padded.abs().max().item()

                assert max_attn_to_padded < 1e-5, (
                    f"Queries are attending to padded key positions! Max attention weight: {max_attn_to_padded:.6e}. "
                    f"Attention weights TO padded key positions should all be zero."
                )

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        # Debug: Check attention output at padded positions
        # if attention_mask_1d is not None:
        #     padded_positions = (attention_mask_1d == 0)
        #     if padded_positions.any():
        #         batch_idx = padded_positions.any(
        #             dim=1).nonzero(as_tuple=True)[0][0].item()
        #         padded_idx = (attention_mask_1d[batch_idx] == 0).nonzero(
        #             as_tuple=True)[0][0].item()

        #         print(f"\n=== ATTENTION OUTPUT DEBUG ===")
        #         print(f"Batch {batch_idx}, Padded position {padded_idx}")
        #         print(
        #             f"Input hidden_states[{batch_idx}, {padded_idx}, :5] = {hidden_states[batch_idx, padded_idx, :5]}")
        #         print(
        #             f"Attention output[{batch_idx}, {padded_idx}, :5] = {attn_output[batch_idx, padded_idx, :5]}")
        #         print(
        #             f"Attention output norm at padded pos: {attn_output[batch_idx, padded_idx].norm().item():.6f}")
        #         print(
        #             f"Input norm at padded pos: {hidden_states[batch_idx, padded_idx].norm().item():.6f}")
        #         print(
        #             f"Are they the same? Max diff: {(attn_output[batch_idx, padded_idx] - hidden_states[batch_idx, padded_idx]).abs().max().item():.6f}")
        #         print(f"=== END ATTENTION OUTPUT DEBUG ===\n")

        return attn_output, attn_weights, past_key_value
