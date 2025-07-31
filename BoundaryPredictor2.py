import torch.nn.utils.rnn
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loss import binomial_loss, hinge_loss
from utils import delete, downsample, weighted_downsample


class BoundaryPredictor2(nn.Module):
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

    def set_prior(self, prior):
        self.prior = prior

    def forward(self, hidden, attention_mask=None, return_boundary_positions=False):
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            # Move normalization to before the projection layer
            self.q_proj_layer(F.normalize(hidden[:, :-1])),
            self.k_proj_layer(F.normalize(hidden[:, 1:]))
        )

        # Append -1 to cos_sim to match the length of hidden
        cos_sim = F.pad(cos_sim, (1, 0), "constant", -1.0)

        probs = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        # probs = torch.sigmoid(cos_sim)
        # probs = F.pad(probs, (1, 0), "constant", 1.0)

        bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=self.temp,
            probs=probs,
        )

        soft_boundaries = bernoulli.rsample()

        hard_boundaries = (soft_boundaries > self.threshold).float()
        hard_boundaries = (
            hard_boundaries - soft_boundaries.detach() + soft_boundaries
        )

        # Manually disable hard boundaries at masked positions after sampling
        if attention_mask is not None:
            masked_positions = (attention_mask == 0).float()
            hard_boundaries = hard_boundaries * (1 - masked_positions)

        # pooled = weighted_downsample(hard_boundaries, cos_sim, hidden)
        pooled = downsample(hard_boundaries, hidden)  # S x B x D
        # pooled = delete(hard_boundaries, hidden)  # S x B x D

        pooled = pooled.transpose(0, 1)

        pooled_attention_mask = attention_mask
        # Update attention mask to match the pooled output size
        if pooled_attention_mask is not None:
            # Keep attention mask values only where hard_boundaries == 1
            pooled_attention_mask = [pooled_attention_mask[b][hard_boundaries[b] == 1]
                                     for b in range(pooled_attention_mask.shape[0])]
            # Pad with zeros so all batch items have the same length
            pooled_attention_mask = torch.nn.utils.rnn.pad_sequence(
                pooled_attention_mask, batch_first=True, padding_value=0)

        # Calculate compression metrics - subtract masked positions
        num_boundaries = hard_boundaries.sum().item()
        total_positions = hard_boundaries.numel()

        if attention_mask is not None:
            masked_count = (attention_mask == 0).sum().item()
            total_positions -= masked_count

        # Calculate loss using adjusted metrics
        loss = self.calc_loss(torch.tensor(num_boundaries, dtype=torch.float, device=hard_boundaries.device),
                              torch.tensor(total_positions, dtype=torch.float, device=hard_boundaries.device))
        self.last_loss = loss  # Store the calculated loss

        if return_boundary_positions:
            # Exclude hard_boundaries at or past the first 0 in pooled_mask for each batch
            filtered_hard_boundaries = []
            for b in range(hard_boundaries.shape[0]):
                mask_row = attention_mask[b]
                hb_row = hard_boundaries[b]
                # Find the first zero in attention_mask
                zero_indices = (mask_row == 0).nonzero(as_tuple=True)[0]
                cutoff = zero_indices[0].item()
                filtered_hard_boundaries.append(hb_row[:cutoff])

            filtered_hard_boundaries = torch.stack(filtered_hard_boundaries)

            return pooled, pooled_attention_mask, loss, num_boundaries, total_positions, filtered_hard_boundaries
        else:
            return pooled, pooled_attention_mask, loss, num_boundaries, total_positions

    def calc_loss(self, num_boundaries, total_positions):
        return binomial_loss(num_boundaries, total_positions, self.prior, self.q_proj_layer.weight.device)
        # binomial = torch.distributions.binomial.Binomial(
        #     preds.size(-1),
        #     probs=torch.Tensor([self.prior]).to(preds.device)
        # )
        # loss_boundaries = -binomial.log_prob(
        #     preds.sum(dim=-1)
        # ).mean() / preds.size(-1)

        # return loss_boundaries
