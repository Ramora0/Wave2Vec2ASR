from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WhisperBoundaryOutput:
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor
    compressed_features: torch.Tensor


@dataclass
class WhisperBoundaryState:
    """
    The state of the whisper boundary predictor.

    Contains
        - [has_seen_features] (batch_size,) bool tensor. Whether that batch element has processed any features yet.
        - [last_feature] (batch_size, d_model) tensor. The last feature of the batch element (used for boundary prediction).
    """

    has_seen_features: torch.Tensor  # (batch_size,)
    last_feature: torch.Tensor  # (batch_size, d_model)


class HNetBoundaryPredictor(nn.Module):
    """
    Boundary predictor for Whisper models that compresses audio features
    by predicting boundaries based on cosine similarity between consecutive features.

    This is adapted from HNet's RoutingModule but tailored for Whisper's audio encoding pipeline.
    """

    def __init__(self, d_model, device=None, dtype=None):
        """
        Args:
            d_model: Dimension of the feature vectors (typically Whisper's encoder dimension)
            device: Device to place the model on
            dtype: Data type for the model parameters
        """
        super().__init__()
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}

        # Linear projections for computing similarity (initialized as identity)
        self.q_proj_layer = nn.Linear(
            d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(
            d_model, d_model, bias=False, **factory_kwargs)

        # Initialize as identity matrices (same as RoutingModule)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model))
            self.k_proj_layer.weight.copy_(torch.eye(d_model))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def forward(self, hidden):
        """
        Forward pass for boundary prediction and feature compression.

        Args:
            hidden: (batch_size, seq_len, d_model) - Encoded audio features from Whisper encoder
            attention_mask: (batch_size, seq_len) - Mask for valid audio frames
            inference_params: Optional state for inference mode

        Returns:
            WhisperBoundaryOutput containing boundary predictions and compressed features
        """
        batch_size, seq_len, d_model = hidden.shape

        # Compute cosine similarity between consecutive features
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hidden[:, 1:]), dim=-1),
        )

        # Convert similarity to boundary probability (lower similarity = higher boundary prob)
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

        # Force boundary probability of the first frame to 1.0 (always a boundary)
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

        # Create binary boundary probabilities [no_boundary_prob, boundary_prob]
        boundary_prob_binary = torch.stack(
            ((1 - boundary_prob), boundary_prob), dim=-1)

        # Select boundaries (argmax for hard decisions)
        selected_idx = torch.argmax(boundary_prob_binary, dim=-1)
        boundary_mask = selected_idx == 1  # (batch_size, seq_len)

        # Get selected probabilities
        selected_probs = boundary_prob_binary.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )  # (batch_size, seq_len, 1)

        # Compress features by selecting only boundary frames
        compressed_features = self._compress_features(
            hidden, boundary_mask)

        return WhisperBoundaryOutput(
            boundary_prob=boundary_prob_binary,  # (batch_size, seq_len, 2)
            boundary_mask=boundary_mask,  # (batch_size, seq_len)
            selected_probs=selected_probs,  # (batch_size, seq_len, 1)
            # (batch_size, num_boundaries, d_model)
            compressed_features=compressed_features,
        )

    def _compress_features(self, hidden, boundary_mask):
        """
        Compress audio features by selecting only boundary frames.

        Args:
            audio_features: (batch_size, seq_len, d_model)
            boundary_mask: (batch_size, seq_len)

        Returns:
            compressed_features: (batch_size, max_boundaries, d_model)
        """
        batch_size, seq_len, d_model = hidden.shape
        device = hidden.device

        # Count boundaries per batch
        num_boundaries = boundary_mask.sum(dim=-1)  # (batch_size,)
        max_boundaries = int(num_boundaries.max())

        if max_boundaries == 0:
            return torch.zeros(batch_size, 0, d_model, device=device, dtype=hidden.dtype)

        # Create indices for gathering boundary features
        boundary_indices = torch.arange(seq_len, device=device)[None, :]
        boundary_indices = boundary_indices + (~boundary_mask).long() * seq_len
        sorted_indices = torch.argsort(boundary_indices, dim=1)

        # Gather boundary features
        compressed_features = torch.gather(
            hidden,
            dim=1,
            index=sorted_indices[:, :max_boundaries,
                                 None].expand(-1, -1, d_model),
        )

        # Create attention mask for compressed features
        compressed_mask = (
            torch.arange(max_boundaries, device=device)[
                None, :] < num_boundaries[:, None]
        )

        # Zero out invalid positions
        compressed_features = compressed_features * \
            compressed_mask.unsqueeze(-1)

        return compressed_features

    def get_compression_ratio(self, boundary_mask):
        """
        Calculate the compression ratio achieved by boundary prediction.

        Args:
            boundary_mask: (batch_size, seq_len)

        Returns:
            compression_ratios: (batch_size,) - ratio of compressed length to original length
        """
        original_lengths = boundary_mask.shape[1]
        compressed_lengths = boundary_mask.sum(dim=-1).float()
        return compressed_lengths / original_lengths
