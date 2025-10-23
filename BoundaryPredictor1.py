import torch.nn.utils.rnn
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loss import binomial_loss, hinge_loss, binomial_loss_from_target_counts
from old_downsample import downsample as old_downsample
from smooth_downsample import downsample_with_smoothed_grad
from utils import downsample, get_sinusoidal_positional_embeddings
# from utils import weighted_downsample  # Optional weighted pooling prototype

# Blend constant: 0.0 = old downsample, 1.0 = new downsample
DOWNSAMPLE_BLEND = 1.0


def explore_placements(probs, confidence=5.0):
    """
    exploration: LOWER = more exploration, HIGHER = less exploration

    exploration = 1.0:  Maximum exploration (very wide Beta)
    exploration = 10.0: Moderate exploration
    exploration = 100.0: Minimal exploration (nearly Bernoulli)
    exploration = ∞:    Exact Bernoulli distribution
    """
    torch.set_printoptions(threshold=float('inf'))
    print(probs[0])
    torch.set_printoptions(threshold=10)
    original_dtype = probs.dtype
    probs_float = probs.float()
    alpha = probs_float * confidence
    beta = (1 - probs_float) * confidence
    # Ensure alpha and beta are > 0 for the Beta distribution
    alpha = torch.clamp(alpha, min=1e-8)
    beta = torch.clamp(beta, min=1e-8)
    beta_dist = torch.distributions.Beta(alpha, beta)
    explored_probs = beta_dist.sample()
    return explored_probs.to(original_dtype)


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

        # Compression scheduling: 0 = every token is a boundary, 1 = only target_boundary_counts boundaries
        self.compression_schedule = 1.0  # Start at max compression by default

        # Store target prior for scheduling (prior will be scheduled from 1.0 to target_prior)
        self.target_prior = prior

        hidden = hidden_dim
        self.dropout = nn.Dropout(p=0.1)
        self.boundary_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

        if init_for_12:
            with torch.no_grad():
                self.boundary_mlp[-1].bias.fill_(-2.5)

    def forward(
        self,
        hidden,
        attention_mask=None,
        target_boundary_counts=None,
        rl=False,
        return_confidence=False,
        return_entropy=False,
    ):
        hidden_for_mlp = self.dropout(hidden)
        logits = self.boundary_mlp(hidden_for_mlp).squeeze(-1)
        probs = torch.sigmoid(logits)

        if rl:
            # RL mode: sample hard boundaries directly
            # if not self.training:
            # Disable exploration during evaluation
            explore_probs = probs
            # else:
            #     # Enable exploration during training
            #     explore_probs = explore_placements(probs, confidence=0.1)
            #     # Print the difference in the probabilities
            #     print(
            #         f"Difference: {(explore_probs - probs).abs().mean().item():.4f}")
            bernoulli = torch.distributions.Bernoulli(probs=explore_probs)
            hard_samples = bernoulli.sample()
            # hard_boundaries = hard_samples
            hard_boundaries = torch.ones_like(hard_samples)
            soft_boundaries = None  # Not used in RL mode
        else:
            # Supervised mode
            if self.training:
                # Use RelaxedBernoulli for differentiable boundaries (STE) during training
                bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                    temperature=self.temp,
                    probs=probs,
                )
                soft_boundaries = bernoulli.rsample()
                hard_samples = (soft_boundaries > self.threshold).float()
            else:
                # During evaluation, threshold probabilities directly
                soft_boundaries = probs
                hard_samples = (probs > 0.5).float()

            hard_boundaries = hard_samples + \
                (soft_boundaries - soft_boundaries.detach())

        if attention_mask is not None:
            hard_boundaries = hard_boundaries * attention_mask
            if soft_boundaries is not None:
                soft_boundaries = soft_boundaries * attention_mask

            # Ensure the last real token is always a boundary
            pad_mask = attention_mask == 0
            if pad_mask.any():
                first_pad_mask = pad_mask & (
                    pad_mask.long().cumsum(dim=1) == 1)
                last_real_mask = torch.roll(
                    first_pad_mask, shifts=-1, dims=1)
                last_real_mask[:, -1] = False
                last_real_mask = last_real_mask.float()
                hard_boundaries = torch.maximum(
                    hard_boundaries, last_real_mask)
                if soft_boundaries is not None:
                    soft_boundaries = torch.maximum(
                        soft_boundaries, last_real_mask)

        pooled = downsample(
            hard_boundaries,
            hidden.transpose(0, 1),
            attention_mask=attention_mask
        )

        pooled = pooled.transpose(0, 1)

        pooled = self._add_positional_embeddings(pooled)

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

        # Apply compression scheduling to target_boundary_counts
        if target_boundary_counts is not None:
            effective_target_counts = self.get_scheduled_target_counts(
                target_boundary_counts, attention_mask)
        else:
            effective_target_counts = None

        log_prob = None
        if rl:
            # For RL: return per-sample boundary losses (B,) and log_prob
            # if effective_target_counts is not None:
            #     loss = self.calc_loss_target_counts_per_item_unreduced(
            #         hard_boundaries, attention_mask, effective_target_counts)
            # else:
            loss = self.calc_example_loss_unreduced(
                hard_boundaries, attention_mask)
            # Cap loss at 1.0 for stable RL training - provides consistent gradient
            # signal regardless of how far from target, allowing natural curriculum
            loss = torch.clamp(loss, max=1.0)
            log_prob = self.compute_log_prob(
                hard_samples, probs, attention_mask)  # (B,)
        else:
            # Normal training: return scalar loss
            # loss = self.calc_loss(num_boundaries_tensor,
            loss = 10 * self.calc_loss_target_counts_per_item(
                hard_boundaries, attention_mask, effective_target_counts)
            # loss = 10 * self.calc_example_loss(
            #     hard_boundaries, attention_mask)
            self.last_loss = loss

        num_boundaries = num_boundaries_tensor.item()
        total_positions = total_positions_tensor.item()

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

    def get_scheduled_target_counts(self, target_boundary_counts, attention_mask=None):
        """
        Compute the effective target boundary counts based on compression schedule.
        Interpolates between max boundaries (sequence length) and target_boundary_counts.

        Args:
            target_boundary_counts: Original target counts (B,) or scalar
            attention_mask: (B, L) attention mask to determine actual sequence lengths

        Returns:
            Effective target counts interpolated based on compression_schedule
        """
        if target_boundary_counts is None:
            return None

        # Determine max boundaries per sample
        if attention_mask is not None:
            # Max boundaries = sequence length for each sample
            max_boundaries = attention_mask.sum(
                dim=1).to(dtype=torch.float32)  # (B,)
        else:
            # If no mask, assume all positions are valid
            # We need the sequence length - this will be batch size dependent
            # For now, use a large number or infer from context
            max_boundaries = target_boundary_counts * \
                12.0  # Placeholder, will be overridden

        # Interpolate: schedule=0 -> max_boundaries, schedule=1 -> target_boundary_counts
        # effective = target + (max - target) * (1 - schedule)
        effective_counts = target_boundary_counts + \
            (max_boundaries - target_boundary_counts) * \
            (1.0 - self.compression_schedule)

        return effective_counts

    def set_prior(self, prior):
        self.prior = prior

    def set_gradient_schedule_alpha(self, alpha):
        """Set the alpha value for gradient scheduling (0.0 to 0.33)"""
        self.gradient_schedule_alpha = float(alpha)

    def set_compression_schedule(self, schedule_value):
        """
        Set the compression schedule value (0.0 to 1.0).
        0.0 = no compression (every token is a boundary)
        1.0 = max compression (only target_boundary_counts boundaries)
        """
        self.compression_schedule = float(schedule_value)

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

    def _add_positional_embeddings(self, x):
        pos_embeds = get_sinusoidal_positional_embeddings(x)
        return x + pos_embeds

    def calc_loss(self, num_boundaries, total_positions):
        scheduled_prior = self.get_scheduled_prior()
        return binomial_loss(num_boundaries, total_positions, scheduled_prior)
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
        scheduled_prior = self.get_scheduled_prior()
        per_example_loss = binomial_loss(
            per_item_boundaries, per_item_totals, scheduled_prior
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
        scheduled_prior = self.get_scheduled_prior()
        per_example_loss = binomial_loss(
            per_item_boundaries, per_item_totals, scheduled_prior
        )
        return per_example_loss  # (B,) - don't reduce

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
