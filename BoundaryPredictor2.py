import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from loss import binomial_loss, binomial_loss_from_target_counts
from old_downsample import downsample as legacy_downsample
from utils import downsample as differentiable_downsample


class BoundaryPredictor2(nn.Module):
    def __init__(self, input_dim, hidden_dim, prior, temp=1, threshold=0.5):
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.threshold = threshold
        self.allow_downsample_gradients = True
        self.downsample_assignment_temp = 5.0
        self.downsample_mask_scale = 5.0

        self.q_proj_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj_layer = nn.Linear(input_dim, input_dim, bias=False)

        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(input_dim))
            self.k_proj_layer.weight.copy_(torch.eye(input_dim))

        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def set_prior(self, prior):
        self.prior = prior

    def set_downsample_gradients(self, enabled: bool):
        self.allow_downsample_gradients = bool(enabled)

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

        pooled_hard = legacy_downsample(hard_boundaries, hidden)

        pooled_soft = differentiable_downsample(
            hard_boundaries,
            hidden,
            assignment_temperature=self.downsample_assignment_temp,
            mask_scale=self.downsample_mask_scale,
        )

        pooled = pooled_hard + (pooled_soft - pooled_soft.detach())

        self._validate_downsample_output(pooled_hard, hard_boundaries)
        pooled = pooled.transpose(0, 1)

        if not self.allow_downsample_gradients:
            pooled = pooled.detach()

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

    def _validate_downsample_output(self, pooled, hard_boundaries, tol=1e-5):
        """Raise if pooled segments disagree with boundary counts."""
        if hard_boundaries.ndim != 2:
            raise RuntimeError(
                "Expected hard_boundaries to be 2D (B x L)."
                f" Got shape {tuple(hard_boundaries.shape)}")

        with torch.no_grad():
            if torch.isnan(pooled).any():
                raise RuntimeError(
                    "Downsample produced NaNs."
                    f" pooled={pooled.detach().cpu()}")

            per_item_segments = hard_boundaries.sum(dim=1)
            max_expected = int(per_item_segments.max().item()
                               ) if per_item_segments.numel() else 0

            if pooled.size(0) != max_expected:
                raise RuntimeError(
                    "Segment count mismatch between pooled output and boundary sums."
                    f" pooled_shape={tuple(pooled.shape)}"
                    f" expected_max_segments={max_expected}"
                    f" per_item_segments={per_item_segments.detach().cpu()}")

            if max_expected == 0:
                if pooled.numel() != 0:
                    raise RuntimeError(
                        "Expected zero pooled vectors but received non-empty tensor."
                        f" pooled={pooled.detach().cpu()}")
                return

            for batch_idx in range(pooled.size(1)):
                expected = int(per_item_segments[batch_idx].item())

                if expected < max_expected:
                    tail = pooled[expected:, batch_idx]
                    if tail.abs().max().item() > tol:
                        raise RuntimeError(
                            "Non-zero pooled vectors found beyond declared boundaries."
                            f" sequence={batch_idx}"
                            f" expected_segments={expected}"
                            f" offending_tail={tail.detach().cpu()}"
                            f" boundaries={hard_boundaries[batch_idx].detach().cpu()}"
                        )

                if expected > 0:
                    active = pooled[:expected, batch_idx]
                    if active.abs().max().item() <= tol:
                        raise RuntimeError(
                            "All active pooled vectors are near zero."
                            f" sequence={batch_idx}"
                            f" expected_segments={expected}"
                            f" active_vectors={active.detach().cpu()}"
                            f" boundaries={hard_boundaries[batch_idx].detach().cpu()}"
                        )
