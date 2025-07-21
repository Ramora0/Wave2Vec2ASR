import torch.nn.utils.rnn
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import delete, downsample


class BoundaryPredictor2(nn.Module):
    def __init__(self, input_dim, hidden_dim, prior, temp=1, threshold=0.5, num_heads=8):
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
        self.v_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.out_proj = nn.Linear(input_dim, 1, bias=False)

        # Initialize projection layers
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # Prevent re-initialization during model loading
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True
        self.v_proj.weight._no_reinit = True
        self.out_proj.weight._no_reinit = True

    def set_prior(self, prior):
        self.prior = prior

    def forward(self, hidden):
        batch_size, seq_len, embed_dim = hidden.shape

        # Project to Q, K, V using full sequence
        Q = self.q_proj(hidden)  # [batch, seq_len, embed_dim]
        K = self.k_proj(hidden)  # [batch, seq_len, embed_dim]
        V = self.v_proj(hidden)  # [batch, seq_len, embed_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        # Shape: [batch, num_heads, seq_len, head_dim]

        # Calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # Shape: [batch, num_heads, seq_len, seq_len]

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        # Shape: [batch, num_heads, seq_len, head_dim]

        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )

        # Calculate boundary probabilities as sum of attention weights
        # Sum attention weights across the key dimension to get boundary score for each position
        boundary_scores = attention_weights.sum(
            dim=-1).mean(dim=1)  # Average across heads
        # Shape: [batch, seq_len]

        # Project to boundary probabilities
        boundary_logits = self.out_proj(
            attended_values).squeeze(-1)  # [batch, seq_len]

        # Combine attention weights and projection for final probabilities
        probs = torch.sigmoid(boundary_logits + boundary_scores)
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

        pooled = downsample(hard_boundaries, hidden)  # S x B x D
        # pooled = delete(hard_boundaries, hidden)  # S x B x D

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
