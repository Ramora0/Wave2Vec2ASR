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

    def set_prior(self, prior):
        self.prior = prior

    def forward(self, hidden):
        batch_size, seq_len, embed_dim = hidden.shape
        print(
            f"Input shape: batch={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}")

        # Add positional embeddings
        positions = torch.arange(seq_len, device=hidden.device).unsqueeze(
            0).expand(batch_size, -1)
        pos_embeddings = self.pos_embedding(positions)
        hidden = hidden + pos_embeddings
        print(f"After adding positional embeddings: {hidden.shape}")

        # Normalize hidden vectors before projection
        hidden = F.normalize(hidden, p=2, dim=-1)

        # Project to Q, K using normalized sequence
        Q = self.q_proj(hidden)  # [batch, seq_len, embed_dim]
        K = self.k_proj(hidden)  # [batch, seq_len, embed_dim]
        print(f"Q shape: {Q.shape}, K shape: {K.shape}")

        # Calculate attention scores (single head)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # Shape: [batch, seq_len, seq_len]
        print(f"Attention scores shape: {attention_scores.shape}")
        print(
            f"Attention scores range: [{attention_scores.min():.4f}, {attention_scores.max():.4f}]")

        # Mask to only allow attention to the previous token (i-1)
        # Create a mask that allows only the lower diagonal with offset -1
        mask = torch.ones(seq_len, seq_len,
                          device=attention_scores.device, dtype=torch.bool)
        # Allow attention from position i to position i-1
        for i in range(1, seq_len):
            mask[i, i-1] = False
        print(f"Mask pattern (first 5x5):\n{(~mask[:5, :5]).int()}")

        # Mask out all other positions (set to 0)
        attention_scores = attention_scores.masked_fill(mask, 0.0)
        print(
            f"Masked attention scores range: [{attention_scores.min():.4f}, {attention_scores.max():.4f}]")

        # Calculate boundary probabilities - sum as normal (only previous token contributes)
        boundary_scores = attention_scores.sum(dim=-1)  # [batch, seq_len]
        print(f"Boundary scores shape: {boundary_scores.shape}")
        print(f"Boundary scores: {boundary_scores[0, :min(10, seq_len)]}")

        # Apply learnable temperature scaling and threshold adjustment to sigmoid
        print(
            f"Sigmoid temperature: {self.sigmoid_temperature.item():.4f}, threshold: {self.sigmoid_threshold.item():.4f}")
        probs = torch.sigmoid(
            (boundary_scores - self.sigmoid_threshold) / self.sigmoid_temperature)
        probs = torch.clamp(probs, min=0.0, max=1.0)
        probs[:, 0] = 1.0  # Set the first position to 1
        print(f"Probabilities: {probs[0, :min(10, seq_len)]}")

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
