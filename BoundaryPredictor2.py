import torch.nn.utils.rnn
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import delete, downsample


class BoundaryPredictor2(nn.Module):
    def __init__(self, input_dim, hidden_dim, prior, temp=1, threshold=0.5, num_heads=1, attention_dropout=0.1):
        """
        input_dim: dimensionality of per-token vectors (D)
        temp: Gumbel-Sigmoid temperature
        num_heads: number of attention heads
        attention_dropout: dropout probability for attention weights
        """
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.threshold = threshold
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.attention_dropout = nn.Dropout(attention_dropout)

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.sigmoid_temperature = nn.Parameter(torch.ones(1))
        self.sigmoid_threshold = nn.Parameter(torch.zeros(1))

        # Positional embeddings
        self.max_position = 4096  # Maximum sequence length supported
        self.pos_embedding = nn.Embedding(self.max_position, input_dim)

        # Multi-head attention projections
        self.q_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, input_dim, bias=False)

        # Initialize projection layers
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)

        # Initialize positional embeddings
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

        # Prevent re-initialization during model loading
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True

        self.sigma = None

    def set_prior(self, prior):
        self.prior = prior

    def forward(self, hidden):
        batch_size, seq_len, embed_dim = hidden.shape

        # Add positional embeddings
        positions = torch.arange(seq_len, device=hidden.device).unsqueeze(
            0).expand(batch_size, -1)
        pos_embeddings = self.pos_embedding(positions)
        hidden_with_pos = hidden + pos_embeddings

        Q = self.q_proj(hidden_with_pos)
        K = self.k_proj(hidden_with_pos)

        attention_matrix = torch.matmul(
            Q, K.transpose(-2, -1))  # [batch, seq_len, seq_len]

        attention_matrix = self.attention_dropout(attention_matrix)

        batch_size, seq_len, _ = attention_matrix.shape

        diagonal_mask = torch.eye(
            seq_len, device=attention_matrix.device, dtype=torch.bool)
        attention_matrix = attention_matrix.masked_fill(diagonal_mask, 0.0)

        cos_sim = attention_matrix.sum(dim=-1)[:, :-1]

        probs = torch.sigmoid(
            (cos_sim + self.sigmoid_threshold) / self.sigmoid_temperature)
        probs = F.pad(probs, (1, 0), "constant", 1.0)
        # All elements are now treated the same; no forced boundary at the first position

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

        # Calculate kept ratio (ratio of tokens that are not boundaries)
        kept_ratio = (1 - hard_boundaries).sum(dim=-1).float() / \
            hard_boundaries.size(-1)

        return pooled, loss, kept_ratio.mean()

    def calc_loss(self, preds):
        binomial = torch.distributions.binomial.Binomial(
            preds.size(-1),
            probs=torch.Tensor([self.prior]).to(preds.device)
        )
        loss_boundaries = -binomial.log_prob(
            preds.sum(dim=-1)
        ).mean() / preds.size(-1)

        return loss_boundaries

    # def update_sigma(self, boundary_rates):
    #     """
    #     Update sigma based on current batch of boundary rates

    #     Args:
    #         boundary_rates: Tensor of boundary rates for current samples
    #     """
    #     self.sigma = torch.std(boundary_rates)
    #     self.beta = self.prior - 3 * self.sigma

    # def calc_loss(self, preds):
    #     """
    #     Implements the paper's proposed loss: max(k/N - β, 0)
    #     where k/N is the boundary rate and β = α - λσ

    #     Args:
    #         preds: Predictions tensor of shape (..., sequence_length)

    #     Returns:
    #         loss: Scalar loss value
    #     """
    #     # Calculate boundary rate k/N for each sample
    #     boundary_rates = preds.sum(dim=-1) / preds.size(-1)  # k/N

    #     # Update sigma and beta based on current batch
    #     self.update_sigma(boundary_rates)

    #     # Apply the modified loss: max(k/N - β, 0)
    #     loss_per_sample = torch.clamp(boundary_rates - self.beta, min=0.0)

    #     # Return mean loss across batch
    #     return loss_per_sample.mean()
