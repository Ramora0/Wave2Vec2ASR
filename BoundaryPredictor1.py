import torch.nn.utils.rnn
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loss import binomial_loss, hinge_loss, binomial_loss_from_target_counts
from old_downsample import downsample as old_downsample
from utils import downsample as new_downsample
# from utils import weighted_downsample  # Optional weighted pooling prototype

# Blend constant: 0.0 = old downsample, 1.0 = new downsample
DOWNSAMPLE_BLEND = 1.0


class BoundaryPredictor1(nn.Module):
    def __init__(self, input_dim, hidden_dim, prior, temp=1, threshold=0.5, init_for_12=True):
        """
        input_dim: dimensionality of per-token vectors (D)
        hidden_dim: hidden size of the MLP
        tau: Gumbel-Sigmoid temperature
        """
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.threshold = threshold
        self.gradient_schedule_alpha = 0.0  # Start with no gradients from downsample

        hidden = hidden_dim
        self.boundary_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

        if init_for_12:
            with torch.no_grad():
                self.boundary_mlp[-1].bias.fill_(+3)

    def set_prior(self, prior):
        self.prior = prior

    def set_gradient_schedule_alpha(self, alpha):
        """Set the alpha value for gradient scheduling (0.0 to 0.33)"""
        self.gradient_schedule_alpha = float(alpha)

    def forward(
        self,
        hidden,
        attention_mask=None,
        target_boundary_counts=None,
        return_log_probs=False,
        return_unreduced_boundary_loss=False,
        return_confidence=False,
        return_entropy=False,
    ):
        logits = self.boundary_mlp(
            hidden).squeeze(-1)
        probs = torch.sigmoid(logits)

        bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=self.temp,
            probs=probs,
        )

        soft_boundaries = bernoulli.rsample()

        if attention_mask is not None:
            soft_boundaries = soft_boundaries * attention_mask

        hard_samples = (soft_boundaries > self.threshold).float()

        if attention_mask is not None:
            hard_samples = hard_samples * attention_mask

            pad_mask = attention_mask == 0
            if pad_mask.any():
                first_pad_mask = pad_mask & (
                    pad_mask.long().cumsum(dim=1) == 1)
                last_real_mask = torch.roll(first_pad_mask, shifts=-1, dims=1)
                last_real_mask[:, -1] = False
                last_real_mask = last_real_mask.float()
                hard_samples = torch.maximum(hard_samples, last_real_mask)
                soft_boundaries = torch.maximum(
                    soft_boundaries, last_real_mask)

        hard_boundaries = (
            hard_samples - soft_boundaries.detach() + soft_boundaries
        )

        if attention_mask is not None:
            hidden_mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            masked_hidden = hidden * hidden_mask
        else:
            masked_hidden = hidden

        # old_downsample expects boundaries (B, L) and hidden (B, L, D)
        # new_downsample expects boundaries (B, L) and hidden (L, B, D)
        # both return (S, B, D)

        # Call old downsample (no gradients) - expects (B, L, D)
        # pooled_old = old_downsample(
        #     hard_boundaries, masked_hidden).to(masked_hidden.dtype)

        # Call new downsample (with gradients) - expects (L, B, D)
        pooled_new = new_downsample(
            hard_boundaries, masked_hidden.transpose(0, 1))

        # Use STE: values from old, gradients from new (scheduled)
        pooled = pooled_new

        # Debug: Check dtypes
        # print("old dtype:", pooled_old.dtype, "new dtype:", pooled_new.dtype)
        # print(masked_hidden.dtype, hidden.dtype)

        # Check if they give the same result (convert to same dtype for comparison)
        # if not torch.allclose(pooled_old, pooled_new, rtol=1e-5, atol=1e-6):
        #     max_diff = (pooled_old - pooled_new).abs().max().item()
        #     raise RuntimeError(
        #         f"Downsample implementations differ! Max difference: {max_diff:.6e}\n"
        #         f"Old shape: {pooled_old.shape}, dtype: {pooled_old.dtype}\n"
        #         f"New shape: {pooled_new.shape}, dtype: {pooled_new.dtype}"
        #     )

        # Blend between old (no grad) and new (with grad)
        # pooled = (1.0 - DOWNSAMPLE_BLEND) * pooled_old + \
        #     DOWNSAMPLE_BLEND * pooled_new

        # transpose to B x S x D
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

        # Compute loss (either reduced scalar or unreduced per-sample)
        if return_unreduced_boundary_loss:
            # For GRPO: return per-sample boundary losses (B,)
            if target_boundary_counts is not None:
                loss = self.calc_loss_target_counts_per_item_unreduced(
                    hard_boundaries, attention_mask, target_boundary_counts)
            else:
                loss = self.calc_example_loss_unreduced(
                    hard_boundaries, attention_mask)
        else:
            # Normal training: return scalar loss
            if target_boundary_counts is not None:
                loss = self.calc_loss_target_counts_overall(
                    hard_boundaries, attention_mask, target_boundary_counts)
            else:
                loss = self.calc_loss(num_boundaries_tensor,
                                      total_positions_tensor)

        # Store the calculated loss (for backward compatibility)
        if not return_unreduced_boundary_loss:
            self.last_loss = loss

        num_boundaries = num_boundaries_tensor.item()
        total_positions = total_positions_tensor.item()

        # Compute log probabilities for RL training (policy gradient)
        log_prob = None
        if return_log_probs:
            log_prob = self.compute_log_prob(
                hard_samples, probs, attention_mask)  # (B,)

        confidence = None
        if return_confidence:
            confidence_map = torch.abs(probs - 0.5)
            if attention_mask is not None:
                mask = attention_mask.to(confidence_map.dtype)
                denom = mask.sum(dim=1).clamp(min=1.0)
                confidence = (confidence_map * mask).sum(dim=1) / denom
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
        return binomial_loss(num_boundaries, total_positions, self.prior)
        # return hinge_loss(preds, self.prior + 0.05, .05) / (64 ** 2)

    def calc_loss_target_counts_overall(self, hard_boundaries, attention_mask, target_boundary_counts):
        device = hard_boundaries.device

        total_boundaries = hard_boundaries.sum().to(dtype=torch.float32)

        if attention_mask is not None:
            total_positions = attention_mask.sum().to(
                device=device, dtype=torch.float32)
        else:
            total_positions = torch.tensor(
                hard_boundaries.numel(),
                device=device,
                dtype=torch.float32,
            )

        target_total = target_boundary_counts.to(
            device=device, dtype=torch.float32).sum()

        clamped_positions = total_positions.clamp(min=1.0)
        target_prob = (
            target_total / clamped_positions).clamp(min=1e-6, max=1 - 1e-6)

        loss = binomial_loss(total_boundaries, clamped_positions, target_prob)
        return loss

    def calc_loss_target_counts_per_item(self, hard_boundaries, attention_mask, target_boundary_counts):
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

    def calc_loss_target_counts_per_item_unreduced(self, hard_boundaries, attention_mask, target_boundary_counts):
        """Return per-sample boundary losses (unreduced) for GRPO."""
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
        return loss_values  # (B,) - don't reduce

    def calc_example_loss_unreduced(self, hard_boundaries, attention_mask=None):
        """Return per-sample boundary losses (unreduced) for GRPO."""
        per_item_boundaries = hard_boundaries.sum(dim=1)
        if attention_mask is not None:
            per_item_totals = attention_mask.sum(dim=1)
        else:
            per_item_totals = torch.full_like(
                per_item_boundaries, hard_boundaries.size(1), dtype=torch.float)

        # Compute loss per example (don't reduce)
        # TEST: Override prior to 1.0 for 1x compression
        # per_example_loss = binomial_loss(
        #     per_item_boundaries, per_item_totals, self.prior
        # )
        per_example_loss = binomial_loss(
            per_item_boundaries, per_item_totals, self.prior
        )
        return per_example_loss  # (B,) - don't reduce

    # ========== RL Methods (for GRPO training) ==========
    # These methods are used only for RL training and don't affect normal training

    def sample_with_log_prob(self, hidden, attention_mask=None):
        """
        Sample boundaries and return log probabilities for RL training.
        This is separate from forward() to not interfere with normal training.

        Args:
            hidden: (B, L, D) hidden states
            attention_mask: (B, L) attention mask

        Returns:
            hard_boundaries: (B, L) sampled boundaries (with STE)
            log_probs: (B,) log probability of the sampled sequence
        """
        logits = self.boundary_mlp(hidden).squeeze(-1)  # (B, L)
        probs = torch.sigmoid(logits)

        # Sample from Bernoulli distribution
        bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=self.temp,
            probs=probs,
        )
        soft_boundaries = bernoulli.rsample()

        if attention_mask is not None:
            soft_boundaries = soft_boundaries * attention_mask

        hard_samples = (soft_boundaries > self.threshold).float()

        if attention_mask is not None:
            hard_samples = hard_samples * attention_mask

            # Ensure last real position is a boundary
            pad_mask = attention_mask == 0
            if pad_mask.any():
                first_pad_mask = pad_mask & (
                    pad_mask.long().cumsum(dim=1) == 1)
                last_real_mask = torch.roll(first_pad_mask, shifts=-1, dims=1)
                last_real_mask[:, -1] = False
                last_real_mask = last_real_mask.float()
                hard_samples = torch.maximum(hard_samples, last_real_mask)

        # Straight-through estimator
        hard_boundaries = (
            hard_samples - soft_boundaries.detach() + soft_boundaries
        )

        # Compute log probability of the sampled boundaries
        # log p(boundaries) = sum_t [b_t * log(p_t) + (1-b_t) * log(1-p_t)]
        log_probs = self.compute_log_prob(hard_samples, probs, attention_mask)

        return hard_boundaries, log_probs

    def compute_log_prob(self, boundaries, probs=None, attention_mask=None):
        """
        Compute log probability of given boundary sequence.
        Used for importance sampling ratio in GRPO.

        Args:
            boundaries: (B, L) boundary decisions (0 or 1)
            probs: (B, L) boundary probabilities. If None, recompute from current policy.
            attention_mask: (B, L) attention mask

        Returns:
            log_probs: (B,) log probability for each item in batch
        """
        if probs is None:
            # Recompute probs from current policy
            # This is needed when computing the ratio in GRPO
            logits = self.boundary_mlp(boundaries).squeeze(-1)
            probs = torch.sigmoid(logits)

        # Clamp probs to avoid log(0)
        probs = torch.clamp(probs, min=1e-8, max=1 - 1e-8)

        probs = probs.to(torch.float32)
        boundaries = boundaries.detach().to(torch.float32)

        log_prob_t = boundaries * torch.log(probs) + \
            (1 - boundaries) * torch.log1p(-probs)

        # Mask out padding positions
        if attention_mask is not None:
            log_prob_t = log_prob_t * attention_mask

        # Sum over sequence length to get log prob for each sequence
        log_probs = log_prob_t.sum(dim=1)  # (B,)

        return log_probs
