from dataclasses import dataclass, field
import math
import torch
from typing import Optional, Tuple, Dict, List
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.modeling_outputs import BaseModelOutput
from torch import nn

from BoundaryPredictor1 import BoundaryPredictor1
from BoundaryPredictor2 import BoundaryPredictor2
from BoundaryPredictor3 import BoundaryPredictor3
from BoundaryPredictor4 import BoundaryPredictor4
from MagnetAttention import MagnetAttention
from MagnetEncoderLayer import MagnetEncoderLayer
from utils import max_pool_attention_mask, remove_positional_embeddings


@dataclass
class ConvConfig:
    """Configuration for ADDITIONAL conv layers after Whisper's conv1/conv2.

    Whisper's pretrained convolutions are always preserved:
    - conv1: stride 2
    - conv2: stride 1
    - Base downsampling: 2x

    Additional convs are appended after these to achieve further downsampling.

    Examples:
        - None (default): uses only Whisper's conv1/conv2 (2x downsampling)
        - ConvConfig.from_strides([2]): 2 × 1 × 2 = 4x downsampling
        - ConvConfig.from_strides([3]): 2 × 1 × 3 = 6x downsampling
        - ConvConfig.from_strides([2, 2]): 2 × 1 × 2 × 2 = 8x downsampling
    """
    layers: List[Tuple[int, int, int]] = field(default_factory=list)  # (out_channels, kernel_size, stride)

    @property
    def total_stride(self) -> int:
        """Total stride from additional convs only."""
        return math.prod(s for _, _, s in self.layers) if self.layers else 1

    @classmethod
    def from_strides(cls, strides: List[int], hidden_dim: int = 768, kernel_size: int = 3) -> "ConvConfig":
        """Create config from just stride values.

        Args:
            strides: List of stride values for additional conv layers
            hidden_dim: Output channels for each conv layer (default: 768)
            kernel_size: Kernel size for each conv layer (default: 3)

        Returns:
            ConvConfig with layers configured from the stride values
        """
        return cls(layers=[(hidden_dim, kernel_size, s) for s in strides])

    def to_dict(self) -> Dict:
        """Serialize config to dictionary for saving."""
        return {"layers": self.layers}

    @classmethod
    def from_dict(cls, d: Dict) -> "ConvConfig":
        """Deserialize config from dictionary."""
        return cls(layers=[tuple(layer) for layer in d.get("layers", [])])


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
    boundary_cv: Optional[float] = None
    boundary_adjacent_pct: Optional[float] = None


class MagnetWhisperEncoder(WhisperEncoder):
    def load_magnet(self, lp, predictor_type="BoundaryPredictor1", conv_config: Optional[ConvConfig] = None):
        self.boundary_predictors = nn.ModuleList(
            [nn.Identity() for _ in range(12)]
        )
        self.compression_ratios = {}
        self.total_boundaries = 0
        self.total_positions = 0
        self.boundary_target_progress = 1.0

        # Setup additional conv layers if configured
        self._setup_additional_convs(conv_config)

        for layer in self.layers:
            layer.__class__ = MagnetEncoderLayer
            layer.self_attn.__class__ = MagnetAttention
            # Initialize RoPE for this attention module
            layer.self_attn.load_magnet(max_position_embeddings=1500, rope_theta=10000.0)

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
            elif predictor_type == "BoundaryPredictor4":
                self.boundary_predictors[layer_idx] = BoundaryPredictor4(
                    768,
                    768,
                    prior_value,
                    1,
                    0.5
                )
            else:
                raise ValueError(
                    f"Unknown predictor_type: {predictor_type}. Supported types are: BoundaryPredictor1, BoundaryPredictor2, BoundaryPredictor3, BoundaryPredictor4")

    def _setup_additional_convs(self, conv_config: Optional[ConvConfig]):
        """Create additional conv layers that come AFTER Whisper's conv1/conv2.

        Args:
            conv_config: Configuration for additional conv layers, or None for default behavior
        """
        self.conv_config = conv_config
        if conv_config is None or not conv_config.layers:
            self.additional_convs = None
            return

        # Input is output of conv2, which has d_model channels (768 for whisper-base)
        in_channels = self.config.d_model
        self.additional_convs = nn.ModuleList()
        for out_channels, kernel_size, stride in conv_config.layers:
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
            self.additional_convs.append(conv)
            in_channels = out_channels

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
        if not isinstance(predictor_module, (BoundaryPredictor1, BoundaryPredictor2, BoundaryPredictor3, BoundaryPredictor4)):
            return hidden_states, attention_mask_1d, 0.0, None, None, None, None, None, target_pointer

        predictor_input = hidden_states
        target_for_predictor = None
        if isinstance(predictor_module, (BoundaryPredictor1, BoundaryPredictor2, BoundaryPredictor4)):
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
        elif isinstance(predictor_module, BoundaryPredictor4):
            # BoundaryPredictor4 supports target counts but not RL mode
            result = predictor_module(
                predictor_input,
                attention_mask_1d,
                target_boundary_counts=target_for_predictor,
                return_confidence=return_boundary_confidence,
                return_entropy=return_entropy,
            )
        else:
            # BoundaryPredictor3 - no target counts, has RL support
            result = predictor_module(
                predictor_input,
                attention_mask_1d,
                rl=boundary_rl,
                return_confidence=return_boundary_confidence,
                return_entropy=return_entropy,
            )
        # BoundaryPredictor2 now returns 10-tuple: (pooled, loss, num_boundaries, total_positions, shortened_mask, log_prob, confidence, entropy, cv, adjacent_pct)
        # BoundaryPredictor4 returns 9-tuple without log_prob: (pooled, loss, num_boundaries, total_positions, shortened_mask, confidence, entropy, cv, adjacent_pct)
        # Others may return fewer values
        # DEBUG: Print tuple length
        if isinstance(predictor_module, BoundaryPredictor2):
            print(
                f"[DEBUG] BP2 result length: {len(result)}, last value: {result[-1]}")
        if isinstance(predictor_module, BoundaryPredictor4):
            # BP4 returns: (pooled, loss, num_boundaries, total_positions, shortened_mask, confidence, entropy, cv, adjacent_pct)
            final_hs_for_layer, current_b_loss, num_boundaries, total_positions, shortened_attention_mask_1d, layer_confidence, layer_entropy, layer_cv, layer_adjacent_pct = result
            layer_log_prob = None
        elif len(result) == 10:
            # BP2 now returns: (pooled, loss, num_boundaries, total_positions, shortened_mask, log_prob, confidence, entropy, cv, adjacent_pct)
            final_hs_for_layer, current_b_loss, num_boundaries, total_positions, shortened_attention_mask_1d, layer_log_prob, layer_confidence, layer_entropy, layer_cv, layer_adjacent_pct = result
        elif len(result) == 9:
            final_hs_for_layer, current_b_loss, num_boundaries, total_positions, shortened_attention_mask_1d, layer_log_prob, layer_confidence, layer_entropy, layer_cv = result
            # BP1/2/3 don't return adjacent_pct (old versions)
            layer_adjacent_pct = None
        elif len(result) == 8:
            final_hs_for_layer, current_b_loss, num_boundaries, total_positions, shortened_attention_mask_1d, layer_log_prob, layer_confidence, layer_entropy = result
            layer_cv = None
            layer_adjacent_pct = None
        elif len(result) == 7:
            final_hs_for_layer, current_b_loss, num_boundaries, total_positions, shortened_attention_mask_1d, layer_log_prob, layer_confidence = result
            layer_entropy = None
            layer_cv = None
            layer_adjacent_pct = None
        else:
            final_hs_for_layer, current_b_loss, num_boundaries, total_positions, shortened_attention_mask_1d, layer_log_prob = result
            layer_confidence = None
            layer_entropy = None
            layer_cv = None
            layer_adjacent_pct = None

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

        return hidden_states, attention_mask_1d, current_b_loss, layer_log_prob, layer_confidence, layer_entropy, layer_cv, layer_adjacent_pct, target_pointer

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

        # Calculate total stride from Whisper convs + additional convs
        base_stride = self.conv1.stride[0] * self.conv2.stride[0]  # 2 × 1 = 2
        additional_stride = getattr(self, 'conv_config', None)
        additional_stride = additional_stride.total_stride if additional_stride else 1
        total_stride = base_stride * additional_stride

        expected_seq_length = self.config.max_source_positions * total_stride
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Always apply Whisper's pretrained convs first
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        # Apply additional convs if configured
        if hasattr(self, 'additional_convs') and self.additional_convs is not None:
            for conv in self.additional_convs:
                inputs_embeds = nn.functional.gelu(conv(inputs_embeds))

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
        total_cv = None
        cv_layers = 0
        total_adjacent_pct = None
        adjacent_pct_layers = 0

        target_matrix: Optional[torch.Tensor] = None
        if target_boundary_counts is not None:
            target_matrix = target_boundary_counts.to(hidden_states.device)

        target_pointer = 0

        if attention_mask is not None:
            # Pool attention mask with total stride (Whisper convs + additional convs)
            attention_mask = max_pool_attention_mask(attention_mask, stride=total_stride)
            attention_mask_1d = attention_mask
        else:
            attention_mask_1d = None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            (
                hidden_states,
                attention_mask_1d,
                current_b_loss,
                layer_log_prob,
                layer_confidence,
                layer_entropy,
                layer_cv,
                layer_adjacent_pct,
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
            # Run boundary predictor BEFORE the layer
            # (
            #     hidden_states,
            #     attention_mask_1d,
            #     current_b_loss,
            #     layer_log_prob,
            #     layer_confidence,
            #     layer_entropy,
            #     layer_cv,
            #     target_pointer,
            # ) = self._apply_boundary_predictor(
            #     idx,
            #     hidden_states,
            #     attention_mask_1d,
            #     target_matrix,
            #     target_pointer,
            #     boundary_rl,
            #     return_boundary_confidence,
            #     return_entropy
            # )

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
            if layer_cv is not None:
                total_cv = layer_cv if total_cv is None else total_cv + layer_cv
                cv_layers += 1
            if layer_adjacent_pct is not None:
                total_adjacent_pct = layer_adjacent_pct if total_adjacent_pct is None else total_adjacent_pct + layer_adjacent_pct
                adjacent_pct_layers += 1

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

        avg_cv = None
        if cv_layers > 0 and total_cv is not None:
            avg_cv = total_cv / cv_layers

        avg_adjacent_pct = None
        if adjacent_pct_layers > 0 and total_adjacent_pct is not None:
            avg_adjacent_pct = total_adjacent_pct / adjacent_pct_layers

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
            boundary_cv=avg_cv,
            boundary_adjacent_pct=avg_adjacent_pct,
        )
