import torch.nn.utils.rnn
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from loss import binomial_loss, hinge_loss, binomial_loss_from_target_counts, repulsion_loss
from utils import downsample, get_sinusoidal_positional_embeddings

# Blend constant: 0.0 = old downsample, 1.0 = new downsample
DOWNSAMPLE_BLEND = 1.0


class BoundaryPredictor4(nn.Module):
    def __init__(self, input_dim, hidden_dim, prior, temp=1, threshold=0.5, init_for_12=True):
        """
        BoundaryPredictor4: Strided convolution-based boundary prediction without RL support.

        Args:
            input_dim: dimensionality of per-token vectors (D)
            hidden_dim: unused, kept for compatibility
            prior: target boundary probability
            temp: Gumbel-Sigmoid temperature
            threshold: decision threshold for hard boundaries
            init_for_12: whether to initialize for 1/12 compression ratio
        """
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.threshold = threshold
        self.input_dim = input_dim

        # Hardcoded convolution parameters - can be changed here if needed
        self.kernel_size = 3
        self.stride = 1
        # Calculate hidden dim to maintain constant FLOPs across different kernel/stride configs
        # Formula: conv_hidden_dim = input_dim // kernel_size * stride
        self.conv_hidden_dim = input_dim // self.kernel_size * self.stride

        # Compression scheduling: 0 = every token is a boundary, 1 = only target_boundary_counts boundaries
        self.compression_schedule = 0.0  # Start at no compression by default

        # Store target prior for scheduling (prior will be scheduled from 1.0 to target_prior)
        self.target_prior = prior

        # Convolutional boundary predictor
        # Conv1d expects (B, C, L) format where C is channels
        self.boundary_conv = nn.Sequential(
            nn.Conv1d(input_dim, self.conv_hidden_dim,
                      kernel_size=self.kernel_size, stride=self.stride),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.conv_hidden_dim, 1, kernel_size=1)
        )

        if init_for_12:
            with torch.no_grad():
                self.boundary_conv[-1].bias.fill_(-2.5)

    def forward(
        self,
        hidden,
        attention_mask=None,
        target_boundary_counts=None,
        return_confidence=False,
        return_entropy=False,
        rl=False,
    ):
        """
        Forward pass for boundary prediction.

        Args:
            hidden: (B, L, D) input sequence
            attention_mask: (B, L) optional attention mask
            target_boundary_counts: (B,) optional target counts per sample
            return_confidence: whether to return confidence scores
            return_entropy: whether to return entropy scores

        Returns:
            9-tuple: (pooled, loss, num_boundaries, total_positions,
                     shortened_attention_mask, confidence, entropy, cv, adjacent_pct)
        """
        batch_size, seq_len, embed_dim = hidden.shape
        device = hidden.device

        # Transpose for Conv1d: (B, L, D) -> (B, D, L)
        hidden_transposed = hidden.transpose(1, 2)

        # Apply strided convolution: (B, D, L) -> (B, 1, L//stride)
        conv_out = self.boundary_conv(hidden_transposed)
        strided_logits = conv_out.squeeze(1)  # (B, L//stride)

        # Create full-length logits tensor initialized to large negative value
        # This ensures non-stride positions have prob ≈ 0 after sigmoid
        full_logits = torch.full((batch_size, seq_len), -10.0,
                                 device=device, dtype=hidden.dtype)

        # Scatter strided outputs to receptive field endpoints
        # Places boundary at the END of each conv window's receptive field
        # For kernel=3, stride=3: windows [0,1,2], [3,4,5], [6,7,8] -> boundaries at 2, 5, 8
        # For kernel=3, stride=2: windows [0,1,2], [2,3,4], [4,5,6] -> boundaries at 2, 4, 6
        num_stride_positions = strided_logits.shape[1]
        for i in range(num_stride_positions):
            # Receptive field end: (kernel_size-1) + i*stride
            target_pos = self.kernel_size - 1 + i * self.stride
            if target_pos < seq_len:
                full_logits[:, target_pos] = strided_logits[:, i]

        logits = full_logits
        probs = torch.sigmoid(logits)

        # Supervised mode only (no RL support)
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
            print(f"[EVAL] Raw hard samples (before mask): {hard_samples[0].nonzero(as_tuple=True)[0].tolist()}")

        hard_boundaries = hard_samples + \
            (soft_boundaries - soft_boundaries.detach())

        if attention_mask is not None:
            hard_boundaries = hard_boundaries * attention_mask
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
                soft_boundaries = torch.maximum(
                    soft_boundaries, last_real_mask)

        # Print final boundary decisions during eval
        if not self.training:
            boundary_positions = hard_boundaries[0].nonzero(as_tuple=True)[0].tolist()
            print(f"[EVAL] Final hard boundaries (after mask): {boundary_positions}")
            if len(boundary_positions) > 1:
                distances = [boundary_positions[i+1] - boundary_positions[i] for i in range(len(boundary_positions)-1)]
                print(f"[EVAL] Boundary distances: {distances}")

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

        if self.training:
            # Calculate repulsion loss
            rep_loss = repulsion_loss(
                hard_boundaries, attention_mask, kernel_size=5)

            # Calculate adjacent boundary penalty: multiply boundaries by shifted version
            # This penalizes boundaries that are immediately next to each other
            shifted_boundaries = torch.roll(hard_boundaries, shifts=1, dims=1)
            # Zero out the first position after shift to avoid wraparound
            shifted_boundaries[:, 0] = 0
            adjacent_penalty = hard_boundaries * shifted_boundaries
            if attention_mask is not None:
                # Only count adjacent boundaries in valid positions
                adjacent_penalty = adjacent_penalty * attention_mask
            adjacent_loss = adjacent_penalty.sum(dim=1).mean()

            # Supervised training: return scalar loss
            binomial = 10 * self.calc_loss_target_counts_per_item(
                hard_boundaries, attention_mask, effective_target_counts)
            # Add repulsion loss scaled by 1/10 (matches the 10x factor on binomial)
            # Add adjacent boundary penalty scaled by 10 (matches binomial scaling)
            loss = binomial  # + rep_loss
            self.last_loss = loss
        else:
            # Don't calculate loss during evaluation
            loss = torch.tensor(0.0, device=hidden.device)

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

        # Compute coefficient of variation for boundary spacing
        cv = self.compute_boundary_cv(hard_boundaries, attention_mask)

        # Compute percentage of adjacent boundaries
        adjacent_pct = self.compute_adjacent_boundary_pct(
            hard_boundaries, attention_mask)

        return (
            pooled,
            loss,
            num_boundaries,
            total_positions,
            shortened_attention_mask,
            confidence,
            entropy,
            cv,
            adjacent_pct,
        )

    def get_scheduled_target_counts(self, target_boundary_counts, attention_mask=None):
        """
        Compute the effective target boundary counts based on a linear compression rate schedule.
        Interpolates the compression rate from 1x to C_max, where C_max is the target compression rate.

        Args:
            target_boundary_counts: Original target counts (B,) or scalar
            attention_mask: (B, L) attention mask to determine actual sequence lengths

        Returns:
            Effective target counts for the current step in the schedule.
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

        # Ensure target_boundary_counts is not zero to avoid division by zero.
        # Add a small epsilon for stability.
        clamped_target_counts = torch.clamp(
            target_boundary_counts.float(), min=1e-8)

        # Calculate the maximum compression rate
        C_max = max_boundaries / clamped_target_counts

        # Linearly interpolate the compression rate from 1x to C_max
        # C(s) = 1 + (C_max - 1) * s
        current_C = 1.0 + (C_max - 1.0) * self.compression_schedule

        # Calculate the effective number of boundaries for the current compression rate
        # E(s) = L / C(s)
        effective_counts = max_boundaries / current_C

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

    def compute_boundary_cv(self, hard_boundaries, attention_mask=None):
        """
        Compute coefficient of variation of inter-boundary distances.

        Returns:
            CV ≈ 0: evenly distributed (max spread)
            CV ≈ 1: random placement (Poisson process)
            CV > 1: clumped together
        """
        batch_cvs = []

        for b in range(hard_boundaries.size(0)):
            boundaries = hard_boundaries[b]
            if attention_mask is not None:
                seq_len = int(attention_mask[b].sum().item())
                boundaries = boundaries[:seq_len]

            # Get positions of boundaries
            boundary_positions = boundaries.nonzero(as_tuple=True)[0]

            if len(boundary_positions) < 2:
                continue  # Need at least 2 boundaries

            # Compute distances between consecutive boundaries
            distances = boundary_positions[1:] - boundary_positions[:-1]
            distances = distances.float()

            # Coefficient of variation: std / mean
            mean_dist = distances.mean()
            std_dist = distances.std()
            cv = std_dist / (mean_dist + 1e-8)

            batch_cvs.append(cv.item())

        if len(batch_cvs) == 0:
            return 0.0

        return sum(batch_cvs) / len(batch_cvs)

    def compute_adjacent_boundary_pct(self, hard_boundaries, attention_mask=None):
        """
        Compute the percentage of boundaries that occur directly next to each other.

        A boundary is considered "adjacent" if it has another boundary at distance 1
        (either immediately before or immediately after it).

        Examples:
            [2, 3, 6, 9]: positions 2 and 3 are both adjacent → 2/4 = 50%
            [2, 3, 4, 7]: positions 2, 3, and 4 are all adjacent → 3/4 = 75%
            [2, 5, 8, 11]: no adjacent boundaries → 0/4 = 0%

        Returns:
            Percentage (0.0 to 1.0) of boundaries that have a neighbor at distance 1.
            0.0 = no adjacent boundaries (well-spaced)
            1.0 = all boundaries are adjacent to another boundary
        """
        batch_pcts = []

        for b in range(hard_boundaries.size(0)):
            boundaries = hard_boundaries[b]
            if attention_mask is not None:
                seq_len = int(attention_mask[b].sum().item())
                boundaries = boundaries[:seq_len]

            # Get positions of boundaries
            boundary_positions = boundaries.nonzero(as_tuple=True)[0]

            if len(boundary_positions) < 2:
                continue  # Need at least 2 boundaries to have adjacency

            # Compute distances between consecutive boundaries
            distances = boundary_positions[1:] - boundary_positions[:-1]

            # Count boundaries that have a neighbor at distance 1
            # A boundary is adjacent if the gap before it OR after it is 1
            adjacent_count = 0
            for i in range(len(boundary_positions)):
                has_left_neighbor = i > 0 and distances[i - 1] == 1
                has_right_neighbor = i < len(distances) and distances[i] == 1
                if has_left_neighbor or has_right_neighbor:
                    adjacent_count += 1

            total_boundaries = len(boundary_positions)
            pct = adjacent_count / total_boundaries if total_boundaries > 0 else 0.0

            batch_pcts.append(pct)

        if len(batch_pcts) == 0:
            return 0.0

        return sum(batch_pcts) / len(batch_pcts)
