import torch
import torch.nn as nn
from typing import Optional
from transformers.cache_utils import EncoderDecoderCache
from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer


class MagnetDecoderLayer(WhisperDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask_1d: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # print(f"LAYER INPUT: type(past_key_value) = {type(past_key_value)}")
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.Tensor`): encoder attention mask - 1D (batch, encoder_seq_len)
                where 1 = valid, 0 = padded. Used as key_mask_1d in cross-attention.
            decoder_attention_mask_1d (`torch.Tensor`): decoder attention mask - 1D (batch, decoder_seq_len)
                where 1 = valid, 0 = padded. Used as query_mask_1d in self-attention and cross-attention.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # For self-attention, only block padded queries from attending
        # Don't block attending TO padded positions (key_mask_1d=None)
        # Apply causal masking to prevent attending to future positions
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            query_mask_1d=decoder_attention_mask_1d,  # Block padded decoder queries
            key_mask_1d=None,  # Allow attending TO padded positions in self-attention
            is_causal=True,  # Apply causal masking for autoregressive decoding
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # For cross-attention, block padded decoder queries and padded encoder keys
            # No causal masking needed (decoder can attend to all encoder positions)
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                query_mask_1d=decoder_attention_mask_1d,  # Block padded decoder queries
                # key_mask_1d=encoder_attention_mask,  # DISABLED ENCODER MASK
                key_mask_1d=None,  # Block attending to padded encoder keys
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 1 of present_key_value tuple
            present_key_value = (
                present_key_value, cross_attn_present_key_value)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
