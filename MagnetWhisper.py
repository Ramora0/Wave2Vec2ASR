from dataclasses import dataclass
import torch
from typing import Optional, Tuple, Union, Dict
from transformers import WhisperForConditionalGeneration, WhisperModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, Seq2SeqModelOutput
from transformers.models.whisper.modeling_whisper import shift_tokens_right
from torch.nn import CrossEntropyLoss
from torch import nn
import os

from BoundaryPredictor1 import BoundaryPredictor1
from BoundaryPredictor2 import BoundaryPredictor2
from BoundaryPredictor3 import BoundaryPredictor3
from MagnetWhisperDecoder import MagnetWhisperDecoder
from utils import max_pool_attention_mask


class MagnetWhisper(WhisperForConditionalGeneration):

    def load_magnet(self, lp, predictor_type="BoundaryPredictor1"):
        self.model.__class__ = MagnetWhisperModel
        self.model.load_magnet(lp, predictor_type)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Check if boundary predictor files exist
        boundary_params_path = os.path.join(
            pretrained_model_name_or_path, "boundary_params.pt")
        boundary_states_path = os.path.join(
            pretrained_model_name_or_path, "boundary_predictors.bin")

        # Load from separate files
        params = torch.load(boundary_params_path, map_location="cpu")
        layer_priors = params["layer_priors"]
        layer_temps = params.get("layer_temps", [])
        layer_thresholds = params.get("layer_thresholds", [])
        layer_types = params.get("layer_types", [])

        # Initialize the magnet component
        model.__class__ = cls

        # If we have layer_types information, reconstruct boundary_predictors based on saved types
        if layer_types:
            # Convert model to MagnetWhisperModel and encoder to MagnetWhisperEncoder
            model.model.__class__ = MagnetWhisperModel
            model.model.encoder.__class__ = MagnetWhisperEncoder

            # Create new ModuleList based on saved types
            model.model.encoder.boundary_predictors = nn.ModuleList(
                [nn.Identity() for _ in range(12)]
            )
            model.model.encoder.compression_ratios = {}
            # Initialize boundary and position counters in encoder
            model.model.encoder.total_boundaries = 0
            model.model.encoder.total_positions = 0

            # Reconstruct each predictor based on saved type information
            layer_types_dict = dict(layer_types)
            layer_priors_dict = dict(layer_priors)

            for idx in range(12):
                predictor_type = layer_types_dict.get(idx, "Identity")
                if predictor_type == "BoundaryPredictor1" and idx in layer_priors_dict:
                    model.model.encoder.boundary_predictors[idx] = BoundaryPredictor1(
                        768,
                        768,
                        layer_priors_dict[idx],
                        1,
                        0.5
                    )
                elif predictor_type == "BoundaryPredictor2" and idx in layer_priors_dict:
                    model.model.encoder.boundary_predictors[idx] = BoundaryPredictor2(
                        768,
                        768,
                        layer_priors_dict[idx],
                        1,
                        0
                    )
                elif predictor_type == "BoundaryPredictor3" and idx in layer_priors_dict:
                    model.model.encoder.boundary_predictors[idx] = BoundaryPredictor3(
                        768,
                        768,
                        layer_priors_dict[idx],
                        1,
                        0.5
                    )
        else:
            # Fallback to old method for backward compatibility
            model.load_magnet(layer_priors, "BoundaryPredictor1")

        # Load boundary predictor states
        boundary_state_dict = torch.load(
            boundary_states_path, map_location="cpu")

        # Load state dict only for BoundaryPredictor instances, not Identity layers
        for idx, boundary_predictor in enumerate(model.model.encoder.boundary_predictors):
            if isinstance(boundary_predictor, (BoundaryPredictor1, BoundaryPredictor2, BoundaryPredictor3)):
                predictor_key = str(idx)
                if predictor_key in boundary_state_dict:
                    boundary_predictor.load_state_dict(
                        boundary_state_dict[predictor_key])

        # Update temperature and threshold for boundary predictors using saved values
        temp_dict = dict(layer_temps)
        threshold_dict = dict(layer_thresholds)

        for idx, boundary_predictor in enumerate(model.model.encoder.boundary_predictors):
            if isinstance(boundary_predictor, (BoundaryPredictor1, BoundaryPredictor2, BoundaryPredictor3)):
                boundary_predictor.temp = temp_dict[idx]
                boundary_predictor.threshold = threshold_dict[idx]

        return model

    def save_pretrained(self, save_directory, *args, **kwargs):
        super().save_pretrained(save_directory, *args, **kwargs)

        # Save boundary predictor states - only for BoundaryPredictor1 instances
        boundary_states_path = os.path.join(
            save_directory, "boundary_predictors.bin")

        # Create a state dict containing only BoundaryPredictor instances
        boundary_state_dict = {}
        for idx, boundary_predictor in enumerate(self.model.encoder.boundary_predictors):
            if isinstance(boundary_predictor, (BoundaryPredictor1, BoundaryPredictor2, BoundaryPredictor3)):
                boundary_state_dict[str(idx)] = boundary_predictor.state_dict()

        torch.save(boundary_state_dict, boundary_states_path)

        # Save boundary predictor parameters
        boundary_params_path = os.path.join(
            save_directory, "boundary_params.pt")

        # Collect parameters from boundary predictors and track their types
        layer_priors = []
        layer_temps = []
        layer_thresholds = []
        layer_types = []  # Track what type each layer is

        for idx, boundary_predictor in enumerate(self.model.encoder.boundary_predictors):
            if isinstance(boundary_predictor, BoundaryPredictor1):
                layer_priors.append((idx, boundary_predictor.prior))
                layer_temps.append((idx, boundary_predictor.temp))
                layer_thresholds.append((idx, boundary_predictor.threshold))
                layer_types.append((idx, "BoundaryPredictor1"))
            elif isinstance(boundary_predictor, BoundaryPredictor2):
                layer_priors.append((idx, boundary_predictor.prior))
                layer_temps.append((idx, boundary_predictor.temp))
                layer_thresholds.append((idx, boundary_predictor.threshold))
                layer_types.append((idx, "BoundaryPredictor2"))
            elif isinstance(boundary_predictor, BoundaryPredictor3):
                layer_priors.append((idx, boundary_predictor.prior))
                layer_temps.append((idx, boundary_predictor.temp))
                layer_thresholds.append((idx, boundary_predictor.threshold))
                layer_types.append((idx, "BoundaryPredictor3"))
            else:
                layer_types.append((idx, "Identity"))

        params = {
            "layer_priors": layer_priors,
            "layer_temps": layer_temps,
            "layer_thresholds": layer_thresholds,
            "layer_types": layer_types,
        }
        torch.save(params, boundary_params_path)

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values=None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`. `sequence_length` should be smaller than or equal to `config.max_target_positions`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # print(
        #     f"[MagnetWhisper] attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")

        model_output = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # Unpack model output
        if len(model_output) == 3:
            outputs, boundary_loss, compression_ratios = model_output
        else:
            outputs, boundary_loss = model_output
            compression_ratios = {}

        # Store compression ratios for logging
        self._compression_ratios = compression_ratios

        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
            loss += boundary_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def get_and_reset_compression_ratio(self):
        """
        Calculate and return the compression ratio, then reset the counters.

        Returns:
            float: Compression ratio (boundaries / total_positions)
        """
        total_boundaries = self.model.encoder.total_boundaries
        total_positions = self.model.encoder.total_positions
        # Reset encoder counters after collecting
        self.model.encoder.total_boundaries = 0
        self.model.encoder.total_positions = 0

        if total_positions == 0:
            compression_ratio = 0.0
        else:
            compression_ratio = total_boundaries / total_positions

        return compression_ratio


class MagnetWhisperModel(WhisperModel):
    def load_magnet(self, lp, predictor_type="BoundaryPredictor1"):
        self.encoder.__class__ = MagnetWhisperEncoder
        self.encoder.load_magnet(lp, predictor_type)

        self.decoder.__class__ = MagnetWhisperDecoder

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values=None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import AutoFeatureExtractor, WhisperModel
         >>> from datasets import load_dataset

         >>> model = WhisperModel.from_pretrained("openai/whisper-base")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
         >>> input_features = inputs.input_features
         >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
         >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
         >>> list(last_hidden_state.shape)
         [1, 2, 512]
         ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print(
        #     f"[MagnetWhisperModel] attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")

        if encoder_outputs is None:
            input_features = self._mask_input_features(
                input_features, attention_mask=attention_mask)

            encoder_outputs = self.encoder(
                input_features,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        # Create cross-attention mask from encoder attention mask
        cross_attention_mask = None
        if hasattr(encoder_outputs, 'encoder_attention_mask') and encoder_outputs.encoder_attention_mask is not None:
            cross_attention_mask = self.encoder._create_cross_attention_mask(
                encoder_outputs.encoder_attention_mask, decoder_input_ids
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=cross_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        ), encoder_outputs.boundary_loss, getattr(encoder_outputs, 'compression_ratios', {})


class MagnetWhisperEncoder(WhisperEncoder):
    # Dont actually use -inf for masking
    def _convert_attention_mask_to_2d(self, attention_mask):
        """
        Convert a 1D attention mask to a 4D attention mask where only positions
        with value 1 can attend to other positions with value 1.

        Args:
            attention_mask: 1D tensor of shape (batch_size, seq_length) with 1s and 0s

        Returns:
            4D attention mask of shape (batch_size, 1, seq_length, seq_length) where
            mask[i, 0, j, k] = 0 if both position j and k should attend to each other, -1e9 otherwise
        """
        if attention_mask is None:
            return None

        batch_size, seq_length = attention_mask.shape

        # Create 2D mask: only positions with 1 can attend to other positions with 1
        # Shape: (batch_size, seq_length, seq_length)
        attention_mask_2d = attention_mask.unsqueeze(
            1) * attention_mask.unsqueeze(2)

        # Convert to attention mask format: 0 for allowed, -1e9 for blocked
        # Where original mask had 1s, we want 0 (allowed attention)
        # Where original mask had 0s, we want -1e9 (blocked attention)
        attention_mask_2d = (1 - attention_mask_2d) * -1e9

        # Add head dimension: (batch_size, 1, seq_length, seq_length)
        attention_mask_4d = attention_mask_2d.unsqueeze(1)

        return attention_mask_4d

    def _create_cross_attention_mask(self, encoder_attention_mask, decoder_input_ids):
        """
        Create a 2D cross-attention mask that allows decoder tokens to attend to 
        non-padded encoder tokens while masking out padded encoder positions.

        Args:
            encoder_attention_mask: 1D tensor of shape (batch_size, encoder_seq_length) 
                                  with 1s for valid tokens and 0s for padding
            decoder_input_ids: tensor of shape (batch_size, decoder_seq_length)

        Returns:
            4D cross-attention mask of shape (batch_size, 1, decoder_seq_length, encoder_seq_length)
            where mask[i, 0, j, k] = 0 if decoder position j can attend to encoder position k, 
            -1e9 if encoder position k is padded and should be masked out
        """
        if encoder_attention_mask is None:
            return None

        batch_size, encoder_seq_length = encoder_attention_mask.shape
        decoder_seq_length = decoder_input_ids.shape[1] if decoder_input_ids is not None else 1

        # Expand encoder mask to cover all decoder positions
        # Shape: (batch_size, decoder_seq_length, encoder_seq_length)
        cross_attention_mask = encoder_attention_mask.unsqueeze(1).expand(
            batch_size, decoder_seq_length, encoder_seq_length
        )

        # Convert to attention mask format: 0 for allowed attention, -1e9 for blocked
        # Where encoder_attention_mask has 1s (valid), we want 0 (allowed attention)
        # Where encoder_attention_mask has 0s (padding), we want -1e9 (blocked attention)
        cross_attention_mask = (1 - cross_attention_mask) * -1e9

        # Add head dimension: (batch_size, 1, decoder_seq_length, encoder_seq_length)
        cross_attention_mask_4d = cross_attention_mask.unsqueeze(1)

        return cross_attention_mask_4d

    def load_magnet(self, lp, predictor_type="BoundaryPredictor1"):
        self.boundary_predictors = nn.ModuleList(
            [nn.Identity() for _ in range(12)]
        )
        self.compression_ratios = {}  # Track compression ratios by layer
        # Initialize boundary and position counters
        self.total_boundaries = 0
        self.total_positions = 0

        for layer_idx, prior_value in lp:
            if predictor_type == "BoundaryPredictor1":
                self.boundary_predictors[layer_idx] = BoundaryPredictor1(
                    768,
                    768,
                    prior_value,
                    1,
                    0.5
                )
            elif predictor_type == "BoundaryPredictor2":
                self.boundary_predictors[layer_idx] = BoundaryPredictor2(
                    768,
                    768,
                    prior_value,
                    1,
                    0
                )
            elif predictor_type == "BoundaryPredictor3":
                self.boundary_predictors[layer_idx] = BoundaryPredictor3(
                    768,
                    768,
                    prior_value,
                    1,
                    0.5
                )
            else:
                raise ValueError(
                    f"Unknown predictor_type: {predictor_type}. Supported types are: BoundaryPredictor1, BoundaryPredictor2, BoundaryPredictor3")

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        expected_seq_length = self.config.max_source_positions * \
            self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        if attention_mask is not None:
            attention_mask = max_pool_attention_mask(attention_mask)

        embed_pos = self.embed_positions.weight

        # print("Embed pos", embed_pos.shape)
        # print("Inputs embeds", inputs_embeds.shape)
        hidden_states = inputs_embeds + embed_pos[:inputs_embeds.shape[1], :]
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        boundary_loss = 0

        # Convert 1D attention mask to 4D if provided
        # print(
        #     f"[MagnetWhisperEncoder] Input attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        attention_mask_4d = self._convert_attention_mask_to_2d(attention_mask)
        # print(
        #     f"[MagnetWhisperEncoder] Converted attention_mask_4d shape: {attention_mask_4d.shape if attention_mask_4d is not None else None}")
        # Keep track of the 1D attention mask for boundary predictors
        attention_mask_1d = attention_mask

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask_4d,  # Use pre-computed 4D attention mask
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask_4d,  # Use pre-computed 4D attention mask
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                predictor_module = self.boundary_predictors[idx]
                if isinstance(predictor_module, (BoundaryPredictor1, BoundaryPredictor2, BoundaryPredictor3)):
                    result = predictor_module(
                        layer_outputs[0], attention_mask_1d)
                    final_hs_for_layer, current_b_loss, num_boundaries, total_positions, shortened_attention_mask_1d = result
                    # Update the 1D attention mask for subsequent boundary predictors
                    attention_mask_1d = shortened_attention_mask_1d
                    # Convert the shortened 1D mask to 4D for subsequent encoder layers
                    attention_mask_4d = self._convert_attention_mask_to_2d(
                        shortened_attention_mask_1d)
                    # Update encoder's counters
                    self.total_boundaries += num_boundaries
                    self.total_positions += total_positions
                    # Track per-layer compression ratio
                    if total_positions > 0:
                        self.compression_ratios[idx] = num_boundaries / \
                            total_positions
                    else:
                        self.compression_ratios[idx] = 0.0
                    boundary_loss += current_b_loss
                    layer_outputs = (final_hs_for_layer,) + layer_outputs[1:]

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return (tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None), 0)
        return MagnetModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states,
            attentions=all_attentions, boundary_loss=boundary_loss,
            compression_ratios=getattr(self, 'compression_ratios', {}),
            encoder_attention_mask=attention_mask_1d
        )


@dataclass
class MagnetModelOutput(BaseModelOutput):
    boundary_loss: Optional[torch.FloatTensor] = None
    compression_ratios: Optional[Dict] = None
    encoder_attention_mask: Optional[torch.LongTensor] = None
