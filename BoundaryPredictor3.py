import torch.nn.utils.rnn
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loss import binomial_loss
from utils import delete, downsample, cross_attention_downsample


class BoundaryPredictor3(nn.Module):
    def __init__(self, input_dim, hidden_dim, prior, temp=1, threshold=0.5):
        """
        input_dim: dimensionality of per-token vectors (D)
        tau: Gumbel-Sigmoid temperature
        """
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.threshold = threshold

        # Linear projections for computing similarity (initialized as identity)
        self.q_proj_layer = nn.Linear(
            input_dim, input_dim, bias=False)
        self.k_proj_layer = nn.Linear(
            input_dim, input_dim, bias=False)

        # Initialize as identity matrices (same as RoutingModule)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(input_dim))
            self.k_proj_layer.weight.copy_(torch.eye(input_dim))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

        # Cross-attention components for downsampling
        self.cross_attn_query = nn.Linear(input_dim, input_dim, bias=False)
        self.cross_attn_key = nn.Linear(input_dim, input_dim, bias=False)
        self.cross_attn_value = nn.Linear(input_dim, input_dim, bias=False)
        self.cross_attn_scale = input_dim ** -0.5

    def set_prior(self, prior):
        self.prior = prior

    def forward(self, hidden):
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            # Move normalization to before the projection layer
            self.q_proj_layer(F.normalize(hidden[:, :-1])),
            self.k_proj_layer(F.normalize(hidden[:, 1:]))
        )

        probs = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        probs = F.pad(probs, (1, 0), "constant", 1.0)

        bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=self.temp,
            probs=probs,
        )

        soft_boundaries = bernoulli.rsample()

        hard_boundaries = (soft_boundaries > self.threshold).float()
        hard_boundaries = (
            hard_boundaries - soft_boundaries.detach() + soft_boundaries
        )

        # Cross-attention based downsampling
        pooled = cross_attention_downsample(
            hard_boundaries, hidden,
            self.cross_attn_query, self.cross_attn_key, self.cross_attn_value,
            self.cross_attn_scale
        )

        pooled = pooled.transpose(0, 1)

        loss = self.calc_loss(hard_boundaries)
        self.last_loss = loss  # Store the calculated loss

        # Calculate compression metrics
        # Total boundaries across all sequences in batch
        num_boundaries = hard_boundaries.sum().item()
        # Total positions across all sequences in batch
        total_positions = hard_boundaries.numel()

        return pooled, loss, num_boundaries, total_positions

    def calc_loss(self, preds):
        return binomial_loss(preds, self.prior)
