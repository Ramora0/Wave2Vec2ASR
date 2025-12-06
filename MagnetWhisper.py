from dataclasses import dataclass
import torch
from typing import Optional, Tuple, Union, Dict
from transformers import WhisperForConditionalGeneration, WhisperModel
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, Seq2SeqModelOutput
from transformers.models.whisper.modeling_whisper import shift_tokens_right
from torch.nn import CrossEntropyLoss
from torch import nn
import os

from BoundaryPredictor1 import BoundaryPredictor1
from BoundaryPredictor2 import BoundaryPredictor2
from BoundaryPredictor3 import BoundaryPredictor3
from BoundaryPredictor4 import BoundaryPredictor4
from MagnetWhisperDecoder import MagnetWhisperDecoder
from MagnetWhisperEncoder import MagnetWhisperEncoder
from MagnetEncoderLayer import MagnetEncoderLayer
from MagnetDecoderLayer import MagnetDecoderLayer
from MagnetAttention import MagnetAttention


@dataclass
class MagnetSeq2SeqModelOutput(Seq2SeqModelOutput):
    """
    Output from MagnetWhisperModel.

    Extends Seq2SeqModelOutput with magnet-specific fields from the encoder.
    Compatible with both standard WhisperEncoder (fields will be None) and
    MagnetWhisperEncoder (fields will contain boundary prediction data).
    """
    boundary_loss: Optional[torch.FloatTensor] = None
    compression_ratios: Optional[Dict] = None
    boundary_log_probs: Optional[torch.FloatTensor] = None
    boundary_confidence: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None
    boundary_cv: Optional[float] = None
    boundary_adjacent_pct: Optional[float] = None


class MagnetWhisper(WhisperForConditionalGeneration):

    def _reset_boundary_loss_tracker(self):
        self._boundary_loss_total = 0.0
        self._boundary_loss_steps = 0

    def set_boundary_target_progress(self, progress: float):
        progress = float(progress)
        progress = max(0.0, min(1.0, progress))
        self.boundary_target_progress = progress
        if hasattr(self.model, "encoder"):
            setattr(self.model.encoder, "boundary_target_progress", progress)

    def load_magnet(self, lp, predictor_type="BoundaryPredictor1"):
        self.model.__class__ = MagnetWhisperModel
        self.model.load_magnet(lp, predictor_type)
        self.set_boundary_target_progress(1.0)
        self._reset_boundary_loss_tracker()

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

        # Convert model to MagnetWhisperModel and encoder to MagnetWhisperEncoder
        model.model.__class__ = MagnetWhisperModel
        model.model.encoder.__class__ = MagnetWhisperEncoder
        model.model.decoder.__class__ = MagnetWhisperDecoder

        # Convert encoder layers to use MagnetAttention (without affecting boundary predictors)
        for layer in model.model.encoder.layers:
            layer.__class__ = MagnetEncoderLayer
            layer.self_attn.__class__ = MagnetAttention
            # Initialize RoPE for encoder self-attention
            layer.self_attn.load_magnet(max_position_embeddings=1500, rope_theta=10000.0)

        # Convert decoder layers to use MagnetAttention
        for layer in model.model.decoder.layers:
            layer.__class__ = MagnetDecoderLayer
            layer.self_attn.__class__ = MagnetAttention
            layer.encoder_attn.__class__ = MagnetAttention
            # Initialize RoPE for decoder self-attention and cross-attention
            layer.self_attn.load_magnet(max_position_embeddings=1500, rope_theta=10000.0)
            layer.encoder_attn.load_magnet(max_position_embeddings=1500, rope_theta=10000.0)

        # Create new ModuleList based on saved types
        model.model.encoder.boundary_predictors = nn.ModuleList(
            [nn.Identity() for _ in range(12)]
        )
        model.model.encoder.compression_ratios = {}
        # Initialize boundary and position counters in encoder
        model.model.encoder.total_boundaries = 0
        model.model.encoder.total_positions = 0
        model.model.encoder.boundary_target_progress = 1.0

        # If we have layer_types information, reconstruct boundary_predictors based on saved types
        if layer_types:
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
                        0.95,
                        init_for_12=False  # Don't initialize bias when loading from pretrained
                    )
                elif predictor_type == "BoundaryPredictor2" and idx in layer_priors_dict:
                    model.model.encoder.boundary_predictors[idx] = BoundaryPredictor2(
                        768,
                        768,
                        layer_priors_dict[idx],
                        1,
                        0.5
                    )
                elif predictor_type == "BoundaryPredictor3" and idx in layer_priors_dict:
                    model.model.encoder.boundary_predictors[idx] = BoundaryPredictor3(
                        768,
                        768,
                        layer_priors_dict[idx],
                        1,
                        0.5
                    )
                elif predictor_type == "BoundaryPredictor4" and idx in layer_priors_dict:
                    model.model.encoder.boundary_predictors[idx] = BoundaryPredictor4(
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
            if isinstance(boundary_predictor, (BoundaryPredictor1, BoundaryPredictor2, BoundaryPredictor3, BoundaryPredictor4)):
                predictor_key = str(idx)
                if predictor_key in boundary_state_dict:
                    boundary_predictor.load_state_dict(
                        boundary_state_dict[predictor_key])

        # Update temperature and threshold for boundary predictors using saved values
        temp_dict = dict(layer_temps)
        threshold_dict = dict(layer_thresholds)

        for idx, boundary_predictor in enumerate(model.model.encoder.boundary_predictors):
            if isinstance(boundary_predictor, (BoundaryPredictor1, BoundaryPredictor2, BoundaryPredictor3, BoundaryPredictor4)):
                boundary_predictor.temp = temp_dict[idx]
                boundary_predictor.threshold = threshold_dict[idx]

        model._reset_boundary_loss_tracker()

        return model

    def save_pretrained(self, save_directory, *args, **kwargs):
        super().save_pretrained(save_directory, *args, **kwargs)

        # Save boundary predictor states - only for BoundaryPredictor1 instances
        boundary_states_path = os.path.join(
            save_directory, "boundary_predictors.bin")

        # Create a state dict containing only BoundaryPredictor instances
        boundary_state_dict = {}
        for idx, boundary_predictor in enumerate(self.model.encoder.boundary_predictors):
            if isinstance(boundary_predictor, (BoundaryPredictor1, BoundaryPredictor2, BoundaryPredictor3, BoundaryPredictor4)):
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
            elif isinstance(boundary_predictor, BoundaryPredictor4):
                layer_priors.append((idx, boundary_predictor.prior))
                layer_temps.append((idx, boundary_predictor.temp))
                layer_thresholds.append((idx, boundary_predictor.threshold))
                layer_types.append((idx, "BoundaryPredictor4"))
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
        target_boundary_counts: Optional[torch.Tensor] = None,
        return_unreduced_loss: Optional[bool] = False,
        boundary_rl: Optional[bool] = False,
        return_boundary_confidence: Optional[bool] = False,
        return_entropy: Optional[bool] = False,
        num_items_in_batch: Optional[int] = None,
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
        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
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
            cache_position=cache_position,
            target_boundary_counts=target_boundary_counts,
            boundary_rl=boundary_rl,
            return_boundary_confidence=return_boundary_confidence,
            return_entropy=return_entropy,
        )

        # Extract magnet-specific attributes if available (for MagnetSeq2SeqModelOutput)
        # or fall back to defaults (for standard Seq2SeqModelOutput)
        boundary_loss = getattr(outputs, 'boundary_loss', None)
        compression_ratios = getattr(outputs, 'compression_ratios', {})
        boundary_log_probs = getattr(outputs, 'boundary_log_probs', None)
        boundary_confidence = getattr(outputs, 'boundary_confidence', None)
        entropy = getattr(outputs, 'entropy', None)
        boundary_cv = getattr(outputs, 'boundary_cv', None)
        boundary_adjacent_pct = getattr(outputs, 'boundary_adjacent_pct', None)

        # Store compression ratios for logging
        self._compression_ratios = compression_ratios

        # Store boundary log probs for RL training (policy gradient)
        # KEEP ON GPU - needed for gradients
        self._boundary_log_probs = boundary_log_probs

        # Store boundary loss for logging (useful for GRPO)
        # KEEP ON GPU initially - will be moved to CPU by trainer
        self._boundary_loss = boundary_loss

        # Store boundary confidence for diagnostics
        # Move to CPU immediately - only used for logging
        self._boundary_confidence = boundary_confidence.cpu(
        ) if boundary_confidence is not None else None

        # Store entropy for RL entropy bonus
        # Move to CPU immediately - only used for reward computation
        self._entropy = entropy.cpu() if entropy is not None else None

        # Store boundary CV for diagnostics
        # Already a scalar, no need to move to CPU
        self._boundary_cv = boundary_cv

        # Store boundary adjacent percentage for diagnostics
        # Already a scalar, no need to move to CPU
        self._boundary_adjacent_pct = boundary_adjacent_pct

        lm_logits = self.proj_out(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            # For RL training, optionally return unreduced loss (per-sample)
            reduction = 'none' if return_unreduced_loss else 'mean'
            loss_fct = CrossEntropyLoss(reduction=reduction)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            asr_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

            if return_unreduced_loss:
                # Reshape to (batch_size, seq_length) and mean over sequence
                batch_size = lm_logits.shape[0]
                seq_length = lm_logits.shape[1]
                asr_loss = asr_loss.view(
                    batch_size, seq_length).mean(dim=1)  # (B,)
                # Add boundary loss (scalar) to each sample
                loss = asr_loss + boundary_loss
            else:
                # Standard mean-reduced loss
                loss = asr_loss + boundary_loss

        if boundary_loss is not None:
            if not hasattr(self, "_boundary_loss_total"):
                self._reset_boundary_loss_tracker()
            # Handle both scalar and per-sample boundary losses
            if boundary_loss.dim() == 0:
                # Scalar loss
                self._boundary_loss_total += boundary_loss.detach().float().item()
            else:
                # Per-sample loss (B,) - take mean for tracking
                self._boundary_loss_total += boundary_loss.detach().float().mean().item()
            self._boundary_loss_steps += 1

        # if not return_dict:
        #     # Note: This path should not be used in normal operation
        #     # We always use return_dict=True with dataclass outputs
        #     raise NotImplementedError(
        #         "return_dict=False is not supported for MagnetWhisper")

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

    def get_and_reset_boundary_loss(self):
        """Return the mean boundary loss since the last reset and clear the accumulator."""
        steps = getattr(self, "_boundary_loss_steps", 0)
        if steps == 0:
            return None
        average_loss = self._boundary_loss_total / steps
        self._reset_boundary_loss_tracker()
        return average_loss


class MagnetWhisperModel(WhisperModel):
    def load_magnet(self, lp, predictor_type="BoundaryPredictor1"):
        self.encoder.__class__ = MagnetWhisperEncoder
        self.encoder.load_magnet(lp, predictor_type)

        self.decoder.__class__ = MagnetWhisperDecoder
        self.decoder.load_magnet()

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
        cache_position: Optional[torch.LongTensor] = None,
        target_boundary_counts: Optional[torch.Tensor] = None,
        boundary_rl: Optional[bool] = False,
        return_boundary_confidence: Optional[bool] = False,
        return_entropy: Optional[bool] = False,
    ) -> MagnetSeq2SeqModelOutput:
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

        encoder_attention_mask = attention_mask

        if encoder_outputs is None:
            input_features = self._mask_input_features(
                input_features, attention_mask=encoder_attention_mask)

            # Build encoder kwargs - add magnet-specific params only for MagnetWhisperEncoder
            encoder_kwargs = {
                'input_features': input_features,
                'attention_mask': encoder_attention_mask,
                'head_mask': head_mask,
                'output_attentions': output_attentions,
                'output_hidden_states': output_hidden_states,
            }

            # Add magnet-specific parameters if encoder supports them
            if isinstance(self.encoder, MagnetWhisperEncoder):
                encoder_kwargs.update({
                    'target_boundary_counts': target_boundary_counts,
                    'boundary_rl': boundary_rl,
                    'return_boundary_confidence': return_boundary_confidence,
                    'return_entropy': return_entropy,
                })

            encoder_outputs = self.encoder(**encoder_kwargs)

        # For MagnetAttention in decoder cross-attention, use 1D encoder attention mask
        cross_attention_mask = None
        if hasattr(encoder_outputs, 'encoder_attention_mask') and encoder_outputs.encoder_attention_mask is not None:
            cross_attention_mask = encoder_outputs.encoder_attention_mask

        decoder_self_attention_mask = decoder_attention_mask

        # Get encoder hidden states - works with both BaseModelOutput and MagnetModelOutput
        encoder_hidden_states = encoder_outputs.last_hidden_state if hasattr(
            encoder_outputs, 'last_hidden_state') else encoder_outputs[0]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_self_attention_mask,
            encoder_attention_mask=cross_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        # Build output - works with both standard BaseModelOutput and custom MagnetModelOutput
        return MagnetSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=getattr(
                encoder_outputs, 'last_hidden_state', None),
            encoder_hidden_states=getattr(
                encoder_outputs, 'hidden_states', None),
            encoder_attentions=getattr(encoder_outputs, 'attentions', None),
            # Magnet-specific fields - None if using standard WhisperEncoder
            boundary_loss=getattr(encoder_outputs, 'boundary_loss', None),
            compression_ratios=getattr(
                encoder_outputs, 'compression_ratios', {}),
            boundary_log_probs=getattr(
                encoder_outputs, 'boundary_log_probs', None),
            boundary_confidence=getattr(
                encoder_outputs, 'boundary_confidence', None),
            entropy=getattr(encoder_outputs, 'entropy', None),
            boundary_cv=getattr(encoder_outputs, 'boundary_cv', None),
            boundary_adjacent_pct=getattr(
                encoder_outputs, 'boundary_adjacent_pct', None),
        )
