import torch.nn.utils.rnn
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loss import binomial_loss, hinge_loss, binomial_loss_from_target_counts
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

    def forward(self, hidden, attention_mask=None, target_boundary_counts=None):
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

        if attention_mask is not None:
            hard_boundaries = hard_boundaries * attention_mask

            pad_mask = attention_mask == 0
            if pad_mask.any():
                first_pad_mask = pad_mask & (
                    pad_mask.long().cumsum(dim=1) == 1)
                last_real_mask = torch.roll(first_pad_mask, shifts=-1, dims=1)
                last_real_mask[:, -1] = False
                hard_boundaries = torch.maximum(
                    hard_boundaries, last_real_mask.float()
                )

        pooled = downsample(hard_boundaries, hidden)  # S x B x D
        # pooled = delete(hard_boundaries, hidden)  # S x B x D

        pooled = pooled.transpose(0, 1)

        shortened_attention_mask = None

        if attention_mask is not None:
            keep_mask = hard_boundaries == 1
            batch_size = attention_mask.shape[0]
            shortened_masks = []

            for b in range(batch_size):
                keep_indices = keep_mask[b].nonzero(as_tuple=True)[0]
                original_mask = attention_mask[b]
                shortened_mask = original_mask[keep_indices]
                shortened_masks.append(shortened_mask)

            shortened_attention_mask = torch.nn.utils.rnn.pad_sequence(
                shortened_masks, batch_first=True, padding_value=0.0)

        num_boundaries_tensor = hard_boundaries.sum()
        if attention_mask is not None:
            total_positions_tensor = attention_mask.sum()
        else:
            total_positions_tensor = torch.tensor(
                hard_boundaries.numel(), device=hard_boundaries.device, dtype=torch.float)

        # loss = self.calc_loss(num_boundaries_tensor, total_positions_tensor)
        if target_boundary_counts is not None:
            loss = self.calc_loss_target_counts(
                hard_boundaries, attention_mask, target_boundary_counts)
        else:
            loss = self.calc_loss(num_boundaries_tensor,
                                  total_positions_tensor)
            raise NotImplementedError("Unimplemented loss calculation")
        # loss = self.calc_example_loss(hard_boundaries, attention_mask)
        self.last_loss = loss  # Store the calculated loss

        num_boundaries = num_boundaries_tensor.item()
        total_positions = total_positions_tensor.item()

        return pooled, loss, num_boundaries, total_positions, shortened_attention_mask

    def calc_loss(self, num_boundaries, total_positions):
        return binomial_loss(num_boundaries, total_positions, self.prior)
        # return hinge_loss(preds, self.prior + 0.05, .05) / (64 ** 2)

    def calc_loss_target_counts(self, hard_boundaries, attention_mask, target_boundary_counts):
        # Encourage boundary counts to match target segment counts via binomial loss.
        device = hard_boundaries.device
        per_item_boundaries = hard_boundaries.sum(dim=1)

        if attention_mask is not None:
            per_item_totals = attention_mask.sum(dim=1)
        else:
            per_item_totals = torch.full(
                (hard_boundaries.size(0),), hard_boundaries.size(1),
                device=device, dtype=torch.float32
            )

        per_item_totals = per_item_totals.to(dtype=torch.float32)
        target_boundary_counts = target_boundary_counts.to(
            device=device, dtype=torch.float32)

        loss_values = binomial_loss_from_target_counts(
            per_item_boundaries.to(dtype=torch.float32),
            per_item_totals,
            target_boundary_counts,
        )
        return loss_values.mean()

    def calc_example_loss(self, hard_boundaries, attention_mask=None):
        per_item_boundaries = hard_boundaries.sum(dim=1)
        if attention_mask is not None:
            per_item_totals = attention_mask.sum(dim=1)
        else:
            per_item_totals = torch.full_like(
                per_item_boundaries, hard_boundaries.size(1), dtype=torch.float)

        # Compute loss per example and normalize by batch size
        per_example_loss = binomial_loss(
            per_item_boundaries, per_item_totals, self.prior
        )
        return per_example_loss.mean()
