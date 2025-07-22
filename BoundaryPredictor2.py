import torch.nn.utils.rnn
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import delete, downsample


class BoundaryPredictor2(nn.Module):
    def __init__(self, input_dim, hidden_dim, prior, temp=1, threshold=0.5, num_heads=1):
        """
        input_dim: dimensionality of per-token vectors (D)
        temp: Gumbel-Sigmoid temperature
        num_heads: number of attention heads
        """
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.threshold = threshold
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        # Multi-head attention projections
        self.q_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, input_dim, bias=False)

        # Initialize projection layers
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)

        # Prevent re-initialization during model loading
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True

    def set_prior(self, prior):
        self.prior = prior

    def forward(self, hidden):
        batch_size, seq_len, embed_dim = hidden.shape

        # Normalize hidden vectors before projection
        hidden = F.normalize(hidden, p=2, dim=-1)

        # Project to Q, K using normalized sequence
        Q = self.q_proj(hidden)  # [batch, seq_len, embed_dim]
        K = self.k_proj(hidden)  # [batch, seq_len, embed_dim]

        # Calculate attention scores (single head)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # Shape: [batch, seq_len, seq_len]

        # Calculate boundary probabilities directly from attention scores with sigmoid
        # Sum attention scores across the key dimension
        boundary_scores = attention_scores.sum(dim=-1)  # [batch, seq_len]
        probs = torch.sigmoid(boundary_scores)
        probs = torch.clamp(probs, min=0.0, max=1.0)

        bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=self.temp,
            probs=probs,
        )

        soft_boundaries = bernoulli.rsample()

        hard_boundaries = (soft_boundaries > self.threshold).float()
        hard_boundaries = (
            hard_boundaries - soft_boundaries.detach() + soft_boundaries
        )

        # pooled = downsample(hard_boundaries, hidden)  # S x B x D
        pooled = delete(hard_boundaries, hidden)  # S x B x D

        pooled = pooled.transpose(0, 1)

        loss = self.calc_loss(hard_boundaries)
        self.last_loss = loss  # Store the calculated loss

        return pooled, loss

    def calc_loss(self, preds):
        binomial = torch.distributions.binomial.Binomial(
            preds.size(-1),
            probs=torch.Tensor([self.prior]).to(preds.device)
        )
        loss_boundaries = -binomial.log_prob(
            preds.sum(dim=-1)
        ).mean() / preds.size(-1)

        return loss_boundaries
