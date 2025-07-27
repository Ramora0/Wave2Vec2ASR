import torch.nn.utils.rnn
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loss import hinge_loss
from utils import downsample


class BoundaryPredictor1(nn.Module):
    def __init__(self, input_dim, hidden_dim, prior, temp=1, threshold=0.5):
        """
        input_dim: dimensionality of per-token vectors (D)
        hidden_dim: hidden size of the MLP
        tau: Gumbel-Sigmoid temperature
        """
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.threshold = threshold

        hidden = hidden_dim
        self.boundary_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def set_prior(self, prior):
        self.prior = prior

    def forward(self, hidden, return_boundary_positions=False):
        # print("Hidden", hidden.shape)
        bs, seq_len, model_dim = hidden.shape
        logits = self.boundary_mlp(
            hidden).squeeze(-1)
        probs = torch.sigmoid(logits)

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

        # Calculate compression metrics
        # Total boundaries across all sequences in batch
        num_boundaries = hard_boundaries.sum().item()
        # Total positions across all sequences in batch
        total_positions = hard_boundaries.numel()

        if return_boundary_positions:
            return pooled, loss, num_boundaries, total_positions, hard_boundaries
        else:
            return pooled, loss, num_boundaries, total_positions

    def calc_loss(self, preds):
        return hinge_loss(preds, self.prior + 0.05, .05)
        # binomial = torch.distributions.binomial.Binomial(
        #     preds.size(-1),
        #     probs=torch.Tensor([self.prior]).to(preds.device)
        # )
        # loss_boundaries = -binomial.log_prob(
        #     preds.sum(dim=-1)
        # ).mean() / preds.size(-1)

        # return loss_boundaries
