import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from loss import binomial_loss, binomial_loss_from_target_counts
from utils import downsample


class BoundaryPredictor2(nn.Module):
    def __init__(self, input_dim, hidden_dim, prior, temp=1, threshold=0.5):
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.threshold = threshold

        self.q_proj_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj_layer = nn.Linear(input_dim, input_dim, bias=False)

        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(input_dim))
            self.k_proj_layer.weight.copy_(torch.eye(input_dim))

        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def set_prior(self, prior):
        self.prior = prior

    def forward(self, hidden, attention_mask=None, target_boundary_counts=None):
        normalized_hidden = F.normalize(hidden, dim=-1)
        q_hidden = self.q_proj_layer(normalized_hidden[:, :-1])
        k_hidden = self.k_proj_layer(normalized_hidden[:, 1:])

        cos_sim = torch.einsum("bld,bld->bl", q_hidden, k_hidden)
        probs = torch.clamp((1 - cos_sim) * 0.5, min=0.0, max=1.0)
        probs = F.pad(probs, (1, 0), value=1.0)

        bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=self.temp,
            probs=probs,
        )

        soft_boundaries = bernoulli.rsample()
        hard_boundaries = (soft_boundaries > self.threshold).float()
        hard_boundaries = hard_boundaries - soft_boundaries.detach() + soft_boundaries

        if attention_mask is not None:
            hard_boundaries = hard_boundaries * attention_mask

            pad_mask = attention_mask == 0
            if pad_mask.any():
                first_pad_mask = pad_mask & (
                    pad_mask.long().cumsum(dim=1) == 1)
                last_real_mask = torch.roll(first_pad_mask, shifts=-1, dims=1)
                last_real_mask[:, -1] = False
                hard_boundaries = torch.maximum(
                    hard_boundaries, last_real_mask.float())

        pooled = downsample(hard_boundaries, hidden)
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

            shortened_attention_mask = pad_sequence(
                shortened_masks, batch_first=True, padding_value=0.0
            )

        num_boundaries_tensor = hard_boundaries.sum()
        if attention_mask is not None:
            total_positions_tensor = attention_mask.sum()
        else:
            total_positions_tensor = torch.tensor(
                hard_boundaries.numel(),
                device=hard_boundaries.device,
                dtype=torch.float,
            )

        if target_boundary_counts is not None:
            loss = self.calc_loss_target_counts(
                hard_boundaries, attention_mask, target_boundary_counts
            )
        else:
            loss = self.calc_loss(num_boundaries_tensor,
                                  total_positions_tensor)
            raise NotImplementedError(
                "Loss without target counts not implemented")

        self.last_loss = loss

        num_boundaries = num_boundaries_tensor.item()
        total_positions = total_positions_tensor.item()

        return pooled, loss, num_boundaries, total_positions, shortened_attention_mask

    def calc_loss(self, num_boundaries, total_positions):
        return binomial_loss(num_boundaries, total_positions, self.prior)

    def calc_loss_target_counts(self, hard_boundaries, attention_mask, target_boundary_counts):
        device = hard_boundaries.device
        per_item_boundaries = hard_boundaries.sum(dim=1)

        if attention_mask is not None:
            per_item_totals = attention_mask.sum(dim=1)
        else:
            per_item_totals = torch.full(
                (hard_boundaries.size(0),),
                hard_boundaries.size(1),
                device=device,
                dtype=torch.float32,
            )

        per_item_totals = per_item_totals.to(dtype=torch.float32)
        target_boundary_counts = target_boundary_counts.to(
            device=device,
            dtype=torch.float32,
        )

        loss_values = binomial_loss_from_target_counts(
            per_item_boundaries.to(dtype=torch.float32),
            per_item_totals,
            target_boundary_counts,
        )
        return loss_values.mean()
