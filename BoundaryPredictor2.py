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

        # Compression scheduling: 0 = every token is a boundary, 1 = only target_boundary_counts boundaries
        self.compression_schedule = 1.0  # Start at max compression by default

        # Store target prior for scheduling (prior will be scheduled from 1.0 to target_prior)
        self.target_prior = prior

        # Boundary loss weight: 0 = loss has no effect, 1 = full loss effect
        self.boundary_loss_weight = 0.0  # Start with no boundary loss

        self.q_proj_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj_layer = nn.Linear(input_dim, input_dim, bias=False)

        with torch.no_grad():
            # Initialize to produce low cosine similarity (high boundary probability)
            # q_proj = I, k_proj = -I gives cos_sim = -1.0 → prob = 1.0 (all boundaries)
            # You can scale k_proj by a factor < 1.0 to get fewer boundaries
            # e.g., k_proj = -0.5 * I gives cos_sim = -0.5 → prob = 0.75
            self.q_proj_layer.weight.copy_(torch.eye(input_dim))
            # Start with moderate boundaries - 0.2 gives ~60% boundary prob
            # This allows some compression while still being permissive initially
            self.k_proj_layer.weight.copy_(-0.5 * torch.eye(input_dim))

        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def set_prior(self, prior):
        self.prior = prior

    def set_compression_schedule(self, schedule_value):
        """
        Set the compression schedule value (0.0 to 1.0).
        0.0 = no compression (every token is a boundary)
        1.0 = max compression (only target_boundary_counts boundaries)
        """
        self.compression_schedule = float(schedule_value)

    def set_boundary_loss_weight(self, weight):
        """
        Set the boundary loss weight (0.0 to 1.0).
        0.0 = boundary loss has no effect on training
        1.0 = boundary loss has full effect
        """
        self.boundary_loss_weight = float(weight)

    def get_scheduled_prior(self):
        """
        Compute the effective prior based on compression schedule using inverse linear interpolation.

        The prior is scheduled such that 1/prior increases linearly from 1.0 to 1/target_prior:
        - When compression_schedule = 0.0: prior = 1.0 (no compression)
        - When compression_schedule = 1.0: prior = target_prior (full compression)

        Returns:
            Effective prior value interpolated based on compression_schedule
        """
        # Inverse linear interpolation: 1/prior scales linearly
        # prior = target_prior / (target_prior + schedule * (1 - target_prior))
        schedule = self.compression_schedule
        target = self.target_prior

        # Handle edge case where target_prior = 1.0
        if abs(target - 1.0) < 1e-8:
            return 1.0

        scheduled_prior = target / (target + schedule * (1.0 - target))
        return scheduled_prior

    def forward(
        self,
        hidden,
        attention_mask=None,
        target_boundary_counts=None,
        return_log_probs=False,
        return_unreduced_boundary_loss=False,
        return_confidence=False,
        return_entropy=False,
        rl=False,
    ):
        normalized_hidden = F.normalize(hidden, dim=-1)
        batch_size = hidden.size(0)
        q_hidden = self.q_proj_layer(normalized_hidden[:, :-1])
        k_hidden = self.k_proj_layer(normalized_hidden[:, 1:])

        cos_sim = torch.einsum("bld,bld->bl", q_hidden, k_hidden)
        probs = torch.clamp((1 - cos_sim) * 0.5, min=0.0, max=1.0)
        probs = F.pad(probs, (0, 1), value=0.0)

        # torch.set_printoptions(threshold=float('inf'))
        # print(probs[0][0:torch.cumsum(
        #     attention_mask[0], dim=0)[-1].long()])
        # torch.set_printoptions(threshold=10)

        bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=self.temp,
            probs=probs,
        )

        soft_boundaries = bernoulli.rsample()
        hard_samples = (soft_boundaries > self.threshold).float()

        if attention_mask is not None:
            soft_boundaries = soft_boundaries * attention_mask
            hard_samples = hard_samples * attention_mask

            pad_mask = attention_mask == 0
            if pad_mask.any():
                first_pad_mask = pad_mask & (
                    pad_mask.long().cumsum(dim=1) == 1)
                last_real_mask = torch.roll(first_pad_mask, shifts=-1, dims=1)
                last_real_mask[:, -1] = False
                boundary_mask = last_real_mask.float()
                hard_samples = torch.maximum(hard_samples, boundary_mask)
                soft_boundaries = torch.maximum(soft_boundaries, boundary_mask)

        hard_boundaries = (
            hard_samples - soft_boundaries.detach() + soft_boundaries
        )

        # Call downsample with smoothed gradients - expects (B, T) and (T, B, C), returns (S, B, C)
        pooled = downsample(
            hard_boundaries, hidden.transpose(0, 1), attention_mask=attention_mask
        )

        # Transpose to B x S x D
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

        # if target_boundary_counts is not None:
        #     per_sample_loss = self.calc_loss_target_counts(
        #         hard_boundaries,
        #         attention_mask,
        #         target_boundary_counts,
        #         reduce=False,
        #     )
        #     loss = per_sample_loss if return_unreduced_boundary_loss else per_sample_loss.mean()
        # else:
        loss = self.calc_loss(num_boundaries_tensor,
                              total_positions_tensor)

        # if target_boundary_counts is not None:
        #     if return_unreduced_boundary_loss:
        #         self.last_loss = per_sample_loss.mean()
        #     else:
        #         self.last_loss = loss

        num_boundaries = num_boundaries_tensor.item()
        total_positions = total_positions_tensor.item()

        log_prob = None
        if return_log_probs:
            probs_clamped = torch.clamp(
                probs, min=1e-8, max=1 - 1e-8).to(torch.float32)
            action_mask = hard_samples.detach().to(torch.float32)
            log_prob_t = action_mask * torch.log(probs_clamped) + \
                (1 - action_mask) * torch.log1p(-probs_clamped)
            if attention_mask is not None:
                log_prob_t = log_prob_t * attention_mask.to(torch.float32)
            log_prob = log_prob_t.sum(dim=1)

        confidence = None
        if return_confidence:
            confidence_map = torch.abs(probs - 0.5)
            if attention_mask is not None:
                mask = attention_mask.to(confidence_map.dtype)
                confidence_map = confidence_map * mask
                denom = mask.sum(dim=1).clamp(min=1.0)
                confidence = confidence_map.sum(dim=1) / denom
            else:
                confidence = confidence_map.mean(dim=1)
            confidence = confidence.detach()

        entropy = None
        if return_entropy:
            probs_clamped = torch.clamp(
                probs, min=1e-8, max=1 - 1e-8).to(torch.float32)
            entropy_map = -(
                probs_clamped * torch.log(probs_clamped)
                + (1.0 - probs_clamped) * torch.log1p(-probs_clamped)
            )
            if attention_mask is not None:
                entropy_map = entropy_map * \
                    attention_mask.to(entropy_map.dtype)
            entropy = entropy_map.sum(dim=1)

        return (
            pooled,
            loss,
            num_boundaries,
            total_positions,
            shortened_attention_mask,
            log_prob,
            confidence,
            entropy,
        )

    def calc_loss(self, num_boundaries, total_positions):
        scheduled_prior = self.get_scheduled_prior()
        loss = binomial_loss(num_boundaries, total_positions, scheduled_prior)
        return loss * self.boundary_loss_weight

    def calc_loss_target_counts(
        self,
        hard_boundaries,
        attention_mask,
        target_boundary_counts,
        reduce=True,
    ):
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
        if reduce:
            return loss_values.mean()
        return loss_values

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
