import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.cache_utils import EncoderDecoderCache
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MagnetEncoderLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_1d: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask_1d (`torch.LongTensor`): 1D attention mask of size `(batch, seq_len)`
                where 1 = valid token, 0 = padded token.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # Store original hidden states at masked positions for verification

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        original_hidden_states = hidden_states.clone(
        ) if attention_mask_1d is not None else None
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            # query_mask_1d=attention_mask_1d,  # DISABLED ENCODER MASK
            query_mask_1d=None,  # Only block padded queries from attending
            key_mask_1d=None,  # Allow attending TO padded positions in self-attention
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

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
