
import torch
import torch.nn as nn
from transformers.models.whisper.modeling_whisper import WhisperDecoder
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from MagnetAttention import MagnetAttention
from MagnetDecoderLayer import MagnetDecoderLayer


class MagnetWhisperDecoder(WhisperDecoder):
    def load_magnet(self):
        """Convert decoder layers to use MagnetAttention for cross-attention padding support."""
        for layer in self.layers:
            layer.__class__ = MagnetDecoderLayer
            layer.self_attn.__class__ = MagnetAttention
            layer.encoder_attn.__class__ = MagnetAttention

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`EncoderDecoderCache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
                Pre-computed hidden-states that can be used to speed up auto-regressive (sequential) decoding. There are
                four sets of pre-computed hidden-states: key and values states in the self-attention blocks (2) and
                in the cross-attention blocks (2). The `past_key_values` are returned when `use_cache=True` is passed or
                when `config.use_cache=True`

                Two formats are allowed:
                - An [`~cache_utils.EncoderDecoderCache`] instance;
                - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
                `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. It is used to update the
                cache in the correct position and to infer the complete sequence length.
        """
        # print(
        #     f"DECODER INPUT: type(past_key_values) = {type(past_key_values)}")
        # print(f"Cache position: {cache_position}")

        # if not hasattr(past_key_values.cross_attention_cache, 'key_cache'):
        #     past_key_values.cross_attention_cache.key_cache = []
        #     past_key_values.cross_attention_cache.value_cache = []

        # if not hasattr(past_key_values.self_attention_cache, 'key_cache'):
        #     past_key_values.self_attention_cache.key_cache = []
        #     past_key_values.self_attention_cache.value_cache = []

        # print(f"Dynamic cache: {dir(past_key_values.cross_attention_cache)}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is not None and not isinstance(past_key_values, EncoderDecoderCache):
            raise TypeError(
                f"past_key_values must be of type EncoderDecoderCache, but is {type(past_key_values)}")

        # print(
        #     f"DECODER PROCESSED: type(past_key_values) = {type(past_key_values)}")

        past_key_values_length = 0
        if cache_position is not None:
            past_key_values_length = cache_position[0]
        elif past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + input_shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(
                0).repeat(input_shape[0], 1)

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(
                input_ids, past_key_values_length=past_key_values_length, position_ids=position_ids
            )
        else:
            positions = self.embed_positions(
                inputs_embeds, past_key_values_length=past_key_values_length, position_ids=position_ids
            )

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)

        # MagnetAttention uses 1D masks directly - no 4D causal mask needed
        decoder_attention_mask_1d = attention_mask

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (
            output_attentions and encoder_hidden_states is not None) else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    decoder_attention_mask_1d,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,  # past_key_value
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    decoder_attention_mask_1d=decoder_attention_mask_1d,
                    layer_head_mask=(
                        head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_values if use_cache else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None
        # if return_legacy_cache:
        #     next_cache = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
