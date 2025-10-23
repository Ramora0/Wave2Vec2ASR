from dataclasses import dataclass
import torch
from typing import Optional, Tuple, Dict
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.modeling_outputs import BaseModelOutput
from torch import nn

from BoundaryPredictor1 import BoundaryPredictor1
from BoundaryPredictor2 import BoundaryPredictor2
from BoundaryPredictor3 import BoundaryPredictor3
from MagnetAttention import MagnetAttention
from MagnetEncoderLayer import MagnetEncoderLayer
from utils import max_pool_attention_mask, remove_positional_embeddings


@dataclass
class MagnetModelOutput(BaseModelOutput):
    """
    Output from MagnetWhisperEncoder.

    Extends BaseModelOutput with magnet-specific fields for boundary prediction.
    When using standard WhisperEncoder, these fields will be None.
    """
    boundary_loss: Optional[torch.FloatTensor] = None
    compression_ratios: Optional[Dict] = None
    encoder_attention_mask: Optional[torch.LongTensor] = None
    boundary_log_probs: Optional[torch.FloatTensor] = None
    boundary_confidence: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


class MagnetWhisperEncoder(WhisperEncoder):
    def load_magnet(self, lp, predictor_type="BoundaryPredictor1"):
        self.boundary_predictors = nn.ModuleList(
            [nn.Identity() for _ in range(12)]
        )
        self.compression_ratios = {}
        self.total_boundaries = 0
        self.total_positions = 0
        self.boundary_target_progress = 1.0

        for layer in self.layers:
            layer.__class__ = MagnetEncoderLayer
            layer.self_attn.__class__ = MagnetAttention

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
                    0.5
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

    def _apply_boundary_predictor(
        self,
        idx,
        hidden_states,
        attention_mask_1d,
        target_matrix,
        target_pointer,
        boundary_rl,
        return_boundary_confidence,
        return_entropy,
    ):
        # hidden_states = remove_positional_embeddings(
        #     hidden_states, self.embed_positions)

        predictor_module = self.boundary_predictors[idx]
        if not isinstance(predictor_module, (BoundaryPredictor1, BoundaryPredictor2, BoundaryPredictor3)):
            return hidden_states, attention_mask_1d, 0.0, None, None, None, target_pointer

        predictor_input = hidden_states
        target_for_predictor = None
        if isinstance(predictor_module, (BoundaryPredictor1, BoundaryPredictor2)):
            if target_matrix is not None and target_pointer < target_matrix.size(0):
                target_for_predictor = target_matrix[target_pointer]
                target_pointer += 1

            if target_for_predictor is not None:
                if attention_mask_1d is not None:
                    per_item_totals = attention_mask_1d.sum(
                        dim=1).to(target_for_predictor.dtype)
                else:
                    per_item_totals = predictor_input.new_full(
                        (predictor_input.size(0),),
                        predictor_input.size(1),
                    ).to(target_for_predictor.dtype)

                safe_totals = per_item_totals.clamp(min=1.0)
                target_counts = target_for_predictor.to(
                    per_item_totals.dtype)
                safe_targets = torch.where(
                    target_counts > 0,
                    target_counts,
                    safe_totals,
                )

                compression_target = safe_totals / safe_targets
                compression_target = compression_target.clamp(
                    min=1.0)

                progress_value = getattr(
                    self, "boundary_target_progress", 1.0)
                progress_tensor = torch.tensor(
                    float(progress_value),
                    device=safe_totals.device,
                    dtype=safe_totals.dtype,
                ).clamp(0.0, 1.0)

                compression_schedule = torch.lerp(
                    torch.ones_like(compression_target),
                    compression_target,
                    progress_tensor,
                )
                target_for_predictor = safe_totals / compression_schedule

                target_for_predictor = torch.minimum(
                    target_for_predictor, per_item_totals)
                target_for_predictor = torch.clamp(
                    target_for_predictor, min=0.0)

            result = predictor_module(
                predictor_input,
                attention_mask_1d,
                target_boundary_counts=target_for_predictor,
                rl=boundary_rl,
                return_confidence=return_boundary_confidence,
                return_entropy=return_entropy,
            )
        else:
            result = predictor_module(
                predictor_input,
                attention_mask_1d,
                rl=boundary_rl,
                return_confidence=return_boundary_confidence,
                return_entropy=return_entropy,
            )
        if len(result) == 8:
            final_hs_for_layer, current_b_loss, num_boundaries, total_positions, shortened_attention_mask_1d, layer_log_prob, layer_confidence, layer_entropy = result
        elif len(result) == 7:
            final_hs_for_layer, current_b_loss, num_boundaries, total_positions, shortened_attention_mask_1d, layer_log_prob, layer_confidence = result
            layer_entropy = None
        else:
            final_hs_for_layer, current_b_loss, num_boundaries, total_positions, shortened_attention_mask_1d, layer_log_prob = result
            layer_confidence = None
            layer_entropy = None

        # The output of the predictor becomes the input for the encoder layer
        hidden_states = final_hs_for_layer
        attention_mask_1d = shortened_attention_mask_1d

        self.total_boundaries += num_boundaries
        self.total_positions += total_positions
        if total_positions > 0:
            self.compression_ratios[idx] = num_boundaries / \
                total_positions
        else:
            self.compression_ratios[idx] = 0.0

        return hidden_states, attention_mask_1d, current_b_loss, layer_log_prob, layer_confidence, layer_entropy, target_pointer

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        target_boundary_counts=None,
        boundary_rl=False,
        return_boundary_confidence=False,
        return_entropy=False,
        return_dict=True,
    ) -> MagnetModelOutput:
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform.
            attention_mask (`torch.Tensor`, *optional*):
                Optional padding mask over the input features.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
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
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        # embed_pos = self.embed_positions.weight
        # hidden_states = inputs_embeds + embed_pos
        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        boundary_loss = 0.0
        total_log_probs = None
        total_confidence = None
        confidence_layers = 0
        total_entropy = None
        entropy_layers = 0

        target_matrix: Optional[torch.Tensor] = None
        if target_boundary_counts is not None:
            target_matrix = target_boundary_counts.to(hidden_states.device)

        target_pointer = 0

        if attention_mask is not None:
            attention_mask = max_pool_attention_mask(attention_mask)
            attention_mask_1d = attention_mask
        else:
            attention_mask_1d = None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # Run boundary predictor BEFORE the layer
            (
                hidden_states,
                attention_mask_1d,
                current_b_loss,
                layer_log_prob,
                layer_confidence,
                layer_entropy,
                target_pointer,
            ) = self._apply_boundary_predictor(
                idx,
                hidden_states,
                attention_mask_1d,
                target_matrix,
                target_pointer,
                boundary_rl,
                return_boundary_confidence,
                return_entropy
            )

            if layer_log_prob is not None:
                if total_log_probs is None:
                    total_log_probs = layer_log_prob
                else:
                    total_log_probs = total_log_probs + layer_log_prob
            if layer_confidence is not None:
                total_confidence = layer_confidence if total_confidence is None else total_confidence + layer_confidence
                confidence_layers += 1
            if layer_entropy is not None:
                total_entropy = layer_entropy if total_entropy is None else total_entropy + layer_entropy
                entropy_layers += 1

            boundary_loss += current_b_loss

            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True

            if to_drop:
                pass
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask_1d,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask_1d,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        avg_confidence = None
        if confidence_layers > 0 and total_confidence is not None:
            avg_confidence = total_confidence / confidence_layers

        entropy = total_entropy if entropy_layers > 0 else None

        return MagnetModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            boundary_loss=boundary_loss,
            compression_ratios=getattr(self, 'compression_ratios', {}),
            encoder_attention_mask=attention_mask_1d,
            boundary_log_probs=total_log_probs,
            boundary_confidence=avg_confidence,
            entropy=entropy,
        )
