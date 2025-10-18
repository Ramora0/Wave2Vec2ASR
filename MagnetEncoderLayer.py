import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.cache_utils import EncoderDecoderCache
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MagnetAttention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_1d: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

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
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
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

        # Debug: Check attention mask values before applying
        if attention_mask is not None and attention_mask_1d is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

            # Check if there are any padded positions
            padded_positions = (attention_mask_1d == 0)
            if padded_positions.any():
                # Sample one padded position to debug
                batch_idx = padded_positions.any(
                    dim=1).nonzero(as_tuple=True)[0][0].item()
                padded_idx = (attention_mask_1d[batch_idx] == 0).nonzero(
                    as_tuple=True)[0][0].item()

                print(f"\n=== ATTENTION MASK DEBUG ===")
                print(f"Batch {batch_idx}, Padded position {padded_idx}")
                print(
                    f"attention_mask_1d[{batch_idx}, {padded_idx}] = {attention_mask_1d[batch_idx, padded_idx].item()}")
                print(f"attention_mask shape: {attention_mask.shape}")
                print(
                    f"causal_mask[{batch_idx}, 0, {padded_idx}, :5] = {causal_mask[batch_idx, 0, padded_idx, :5]}")
                print(
                    f"causal_mask[{batch_idx}, 0, :5, {padded_idx}] = {causal_mask[batch_idx, 0, :5, padded_idx]}")
                print(
                    f"attn_weights before mask [{batch_idx}, 0, {padded_idx}, :5] = {attn_weights[batch_idx, 0, padded_idx, :5]}")

            attn_weights = attn_weights + causal_mask

            if padded_positions.any():
                print(
                    f"attn_weights after mask [{batch_idx}, 0, {padded_idx}, :5] = {attn_weights[batch_idx, 0, padded_idx, :5]}")
                print(
                    f"MAX attn_weights after mask (padded row): {attn_weights[batch_idx, 0, padded_idx, :].max().item()}")
                print(
                    f"MIN attn_weights after mask (padded row): {attn_weights[batch_idx, 0, padded_idx, :].min().item()}")
                print(
                    f"Are all values in padded row -1e9 or similar? Check if max ≈ min")

                # Check for positions that are NOT -1e9 in the padded row
                padded_row = attn_weights[batch_idx, 0, padded_idx, :]
                non_masked = (padded_row > -1e8)  # Should all be False if properly masked
                if non_masked.any():
                    non_masked_positions = non_masked.nonzero(as_tuple=True)[0]
                    print(f"\n!!! FOUND NON-MASKED POSITIONS IN PADDED ROW !!!")
                    print(f"Positions that are NOT -1e9: {non_masked_positions[:10].tolist()}")
                    print(f"Values at those positions: {padded_row[non_masked_positions[:10]]}")
                    print(f"attention_mask_1d values at those positions: {attention_mask_1d[batch_idx, non_masked_positions[:10]]}")

                print(f"=== END DEBUG ===\n")

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Debug: Check attention weights after softmax
        if attention_mask_1d is not None:
            padded_positions = (attention_mask_1d == 0)
            if padded_positions.any():
                batch_idx = padded_positions.any(dim=1).nonzero(as_tuple=True)[0][0].item()
                padded_idx = (attention_mask_1d[batch_idx] == 0).nonzero(as_tuple=True)[0][0].item()

                print(f"\n=== AFTER SOFTMAX ===")
                print(f"attn_weights after softmax [{batch_idx}, 0, {padded_idx}, :5] = {attn_weights[batch_idx, 0, padded_idx, :5]}")
                print(f"attn_weights after softmax [{batch_idx}, 0, :5, {padded_idx}] = {attn_weights[batch_idx, 0, :5, padded_idx]}")
                print(f"Max attn from padded position: {attn_weights[batch_idx, :, padded_idx, :].max().item()}")
                print(f"Max attn to padded position: {attn_weights[batch_idx, :, :, padded_idx].max().item()}")
                print(f"=== END AFTER SOFTMAX ===\n")

        # Verify that padded positions have zero attention weights
        if attention_mask_1d is not None:
            # attention_mask_1d shape: (batch, seq_len)
            # attn_weights shape: (batch, num_heads, tgt_len, src_len)
            padded_positions = (attention_mask_1d == 0)  # (batch, seq_len)

            if padded_positions.any():
                # Check attention weights FROM padded positions (rows should be zero)
                # Expand to (batch, 1, tgt_len, 1) for broadcasting
                padded_queries = padded_positions.unsqueeze(
                    1).unsqueeze(-1)  # (batch, 1, seq_len, 1)
                attn_from_padded = attn_weights * padded_queries  # Zero out non-padded positions
                max_attn_from_padded = attn_from_padded.abs().max().item()

                assert max_attn_from_padded < 1e-5, (
                    f"Padded positions are attending to others! Max attention weight: {max_attn_from_padded:.6e}. "
                    f"Attention weights FROM padded query positions should all be zero."
                )

                # Check attention weights TO padded positions (columns should be zero)
                # Expand to (batch, 1, 1, src_len) for broadcasting
                padded_keys = padded_positions.unsqueeze(
                    1).unsqueeze(1)  # (batch, 1, 1, seq_len)
                attn_to_padded = attn_weights * padded_keys  # Zero out non-padded positions
                max_attn_to_padded = attn_to_padded.abs().max().item()

                assert max_attn_to_padded < 1e-5, (
                    f"Non-padded positions are attending to padded positions! Max attention weight: {max_attn_to_padded:.6e}. "
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

        return attn_output, attn_weights, past_key_value


class MagnetEncoderLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_mask_1d: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # Store original hidden states at masked positions for verification
        original_hidden_states = hidden_states.clone(
        ) if attention_mask_1d is not None else None

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attention_mask_1d=attention_mask_1d,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value)

        # Verify that masked positions haven't been changed
        if original_hidden_states is not None and attention_mask_1d is not None:
            # Create mask for positions that should be unchanged (where mask_1d == 0)
            masked_positions = (attention_mask_1d == 0)  # (batch, seq_len)
            if masked_positions.any():
                # Expand mask to match hidden states dimensions for indexing
                mask_expanded = masked_positions.unsqueeze(
                    -1).expand_as(original_hidden_states)
                original_masked = original_hidden_states[mask_expanded]
                current_masked = hidden_states[mask_expanded]
                max_diff = (original_masked -
                            current_masked).abs().max().item()
                assert max_diff < 1e-5, (
                    f"Masked positions were modified! Max difference: {max_diff:.6e}. "
                    f"Hidden states at positions masked by attention_mask_1d should not change."
                )

        # Verify that non-padded tokens produce the same output as if run in isolation
        # This ensures padded tokens aren't affecting non-padded tokens through attention
        # if original_hidden_states is not None and attention_mask_1d is not None:
        #     # Pick the first sample that has padding
        #     for batch_idx in range(attention_mask_1d.shape[0]):
        #         mask_1d = attention_mask_1d[batch_idx]
        #         num_valid = mask_1d.sum().item()
        #         total_len = mask_1d.shape[0]

        #         # Only test if there's actual padding
        #         if num_valid < total_len and num_valid > 0:
        #             # Extract non-padded portion
        #             valid_hidden = original_hidden_states[batch_idx:batch_idx+1, :int(
        #                 num_valid), :]

        #             # Run attention on just the valid portion (no mask needed)
        #             valid_residual = valid_hidden
        #             valid_hidden = self.self_attn_layer_norm(valid_hidden)
        #             valid_hidden, _, _ = self.self_attn(
        #                 hidden_states=valid_hidden,
        #                 attention_mask=None,  # No masking needed for valid-only sequence
        #                 layer_head_mask=layer_head_mask,
        #                 output_attentions=False,
        #             )
        #             valid_hidden = nn.functional.dropout(
        #                 valid_hidden, p=self.dropout, training=self.training)
        #             valid_hidden = valid_residual + valid_hidden

        #             # Run feedforward
        #             valid_residual = valid_hidden
        #             valid_hidden = self.final_layer_norm(valid_hidden)
        #             valid_hidden = self.activation_fn(self.fc1(valid_hidden))
        #             valid_hidden = nn.functional.dropout(
        #                 valid_hidden, p=self.activation_dropout, training=self.training)
        #             valid_hidden = self.fc2(valid_hidden)
        #             valid_hidden = nn.functional.dropout(
        #                 valid_hidden, p=self.dropout, training=self.training)
        #             valid_hidden = valid_residual + valid_hidden

        #             if valid_hidden.dtype == torch.float16:
        #                 clamp_value = torch.finfo(
        #                     valid_hidden.dtype).max - 1000
        #                 valid_hidden = torch.clamp(
        #                     valid_hidden, min=-clamp_value, max=clamp_value)

        #             # Compare with the non-padded portion from the full batch
        #             batch_valid = hidden_states[batch_idx:batch_idx +
        #                                         1, :int(num_valid), :]
        #             max_diff = (valid_hidden - batch_valid).abs().max().item()

        #             assert max_diff < 1e-4, (
        #                 f"Non-padded tokens affected by padding! Batch {batch_idx}, "
        #                 f"max difference: {max_diff:.6e}. "
        #                 f"Valid tokens ({int(num_valid)}/{total_len}) should produce identical "
        #                 f"results whether run in isolation or with padding masked."
        #             )

        #             # Only test one sample per forward pass to avoid performance impact
        #             break

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
