import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from loss import binomial_loss, binomial_loss_from_target_counts, binomial_loss_from_target_counts_flexible
from utils import downsample, get_sinusoidal_positional_embeddings, common


class UNet1D(nn.Module):
    """
    Efficient 1-level UNet for sequence segmentation.
    Stays in (B, C, L) format throughout to avoid expensive transposes.

    Args:
        input_dim: Input channel dimension (768)
        hidden_dim: Bottleneck dimension (384)
        dropout: Dropout probability (0.1)
    """
    def __init__(self, input_dim=768, hidden_dim=384, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder: downsample by 2x
        self.encoder_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.encoder_norm = nn.BatchNorm1d(hidden_dim)

        # Bottleneck: same size
        self.bottleneck_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bottleneck_norm = nn.BatchNorm1d(hidden_dim)

        # Decoder: upsample by 2x
        self.decoder_conv = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.decoder_norm1 = nn.BatchNorm1d(input_dim)  # After concat: hidden_dim + hidden_dim = input_dim

        # Refine: smooth the output
        self.refine_conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
        self.decoder_norm2 = nn.BatchNorm1d(input_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize final layer with small weights to start near identity
        with torch.no_grad():
            self.refine_conv.weight.normal_(0, 0.01)
            self.refine_conv.bias.zero_()

    def forward(self, x):
        """
        Args:
            x: (B, L, D) input sequence

        Returns:
            output: (B, L, D) processed sequence
        """
        batch_size, seq_len, dim = x.shape

        # Transpose once to (B, D, L) for Conv1d, keep in this format throughout
        x = x.transpose(1, 2)  # (B, D, L)

        # Encoder
        enc = self.encoder_conv(x)  # (B, hidden_dim, L//2)
        enc = self.encoder_norm(enc)
        enc = F.gelu(enc)
        enc = self.dropout(enc)
        enc_skip = enc  # Save for skip connection

        # Bottleneck
        bottle = self.bottleneck_conv(enc)  # (B, hidden_dim, L//2)
        bottle = self.bottleneck_norm(bottle)
        bottle = F.gelu(bottle)
        bottle = self.dropout(bottle)

        # Decoder upsample
        dec = self.decoder_conv(bottle)  # (B, hidden_dim, ~L)

        # Trim/pad decoder output to match skip connection length
        if dec.shape[2] != enc_skip.shape[2]:
            if dec.shape[2] > enc_skip.shape[2]:
                dec = dec[:, :, :enc_skip.shape[2]]
            else:
                dec = F.pad(dec, (0, enc_skip.shape[2] - dec.shape[2]))

        # Concatenate skip connection
        dec = torch.cat([dec, enc_skip], dim=1)  # (B, input_dim, L//2)
        dec = self.decoder_norm1(dec)
        dec = F.gelu(dec)
        dec = self.dropout(dec)

        # Upsample to original length and refine
        if dec.shape[2] != seq_len:
            # Only interpolate if necessary
            dec = F.interpolate(dec, size=seq_len, mode='linear', align_corners=False)

        out = self.refine_conv(dec)  # (B, input_dim, L)
        out = self.decoder_norm2(out)

        # Transpose back to (B, L, D)
        out = out.transpose(1, 2)

        return out


class BoundaryPredictor3(nn.Module):
    def __init__(self, input_dim, hidden_dim, prior, temp=1, threshold=0.5, max_positions=1500, use_attention_pooling=True):
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

        # UNet for segmentation features
        self.unet = UNet1D(input_dim, hidden_dim=384)

        # MLP layers for Q and K paths
        self.q_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.k_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.q_proj_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj_layer = nn.Linear(input_dim, input_dim, bias=False)

        self.similarity_bias = nn.Parameter(torch.tensor(-0.126))

        with torch.no_grad():
            # Initialize with identity matrices
            self.q_proj_layer.weight.copy_(torch.eye(input_dim))
            self.k_proj_layer.weight.copy_(torch.eye(input_dim))

        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

        self.dropout = nn.Dropout(p=0.1)

        # Attention pooling parameters
        self.use_attention_pooling = use_attention_pooling

        if self.use_attention_pooling:
            # Multi-head attention configuration
            self.num_heads = 8
            self.head_dim = input_dim // self.num_heads
            assert input_dim % self.num_heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({self.num_heads})"

            # Learned query vector (shared across all segments)
            self.learned_query = nn.Parameter(torch.randn(input_dim))

            # Key and Value projections
            self.pool_key = nn.Linear(input_dim, input_dim, bias=False)
            self.pool_value = nn.Linear(input_dim, input_dim, bias=False)

            # Output projection after pooling (combines heads)
            self.pool_output = nn.Linear(input_dim, input_dim, bias=False)

            # LayerNorm for stabilizing attention inputs
            self.pool_layernorm = nn.LayerNorm(input_dim)

            # Scaling factor for attention scores (per head)
            self.pool_scale = self.head_dim ** -0.5

            # Initialize projections as identity matrices
            with torch.no_grad():
                self.pool_key.weight.copy_(torch.eye(input_dim))
                self.pool_value.weight.copy_(torch.eye(input_dim))
                self.pool_output.weight.copy_(torch.eye(input_dim))

            self.pool_key.weight._no_reinit = True
            self.pool_value.weight._no_reinit = True
            self.pool_output.weight._no_reinit = True

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

    def set_pooling_method(self, use_attention: bool):
        """
        Switch between weighted mean and attention pooling.

        Args:
            use_attention: If True, use attention pooling; if False, use weighted mean pooling

        Raises:
            ValueError: If attention pooling is requested but parameters were not initialized
        """
        if use_attention and not hasattr(self, 'learned_query'):
            raise ValueError(
                "Attention pooling parameters not initialized. "
                "Create model with use_attention_pooling=True"
            )
        self.use_attention_pooling = use_attention

    def get_scheduled_prior(self):
        """
        Compute the effective prior based on compression schedule using inverse linear interpolation.

        The prior is scheduled such that 1/prior increases linearly from 1.0 to 1/target_prior:
        - When compression_schedule = 0.0: prior = 1.0 (no compression)
        - When compression_schedule = 1.0: prior = target_prior (full compression)

        Returns:
            Effective prior value interpolated based on compression_schedule
        """
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

    def _weighted_mean_pooling(self, boundaries, hidden, attention_mask=None):
        """Weighted mean pooling using segment assignment normalization."""
        pooled = downsample(boundaries, hidden.transpose(0, 1), attention_mask=attention_mask)
        return pooled.transpose(0, 1)  # B x S x D

    def _attention_pooling(self, boundaries, hidden, attention_mask=None):
        """
        Multi-head attention-based pooling using query matrix applied to boundary positions.

        Args:
            boundaries: (B, L) - binary boundary indicators
            hidden: (B, L, D) - hidden states
            attention_mask: (B, L) - attention mask

        Returns:
            pooled: (B, S, D) - pooled segment representations
        """
        batch_size, seq_len, hidden_dim = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        # Step 1: Create segment assignment matrix using existing logic
        foo = common(boundaries)  # B x L x S

        if foo is None:
            # No boundaries found
            return torch.empty(batch_size, 0, hidden_dim, device=device, dtype=dtype)

        max_segments = foo.size(2)  # S

        # Step 2: Create binary segment mask (B x L x S)
        # foo == 0 indicates token belongs to segment
        segment_mask = (foo == 0).float()  # B x L x S

        if attention_mask is not None:
            # Apply attention mask: (B, L, 1) * (B, L, S) -> (B, L, S)
            segment_mask = segment_mask * attention_mask.unsqueeze(-1)

        # Step 3: Use learned query vector for all segments
        # Expand learned query to (B, S, D) - same query for all segments in all batches
        queries = self.learned_query.unsqueeze(0).unsqueeze(0).expand(batch_size, max_segments, -1)  # (B, S, D)

        # Step 4: Reshape queries for multi-head attention
        queries = queries.view(batch_size, max_segments, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, head_dim)

        # Step 5: Apply LayerNorm before projecting to keys and values
        hidden_normed = self.pool_layernorm(hidden)  # (B, L, D)

        # Step 6: Project to keys and values and reshape for multi-head
        keys = self.pool_key(hidden_normed)      # (B, L, D)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, head_dim)

        values = self.pool_value(hidden_normed)  # (B, L, D)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, head_dim)

        # Step 7: Compute attention scores: queries @ keys
        # queries: (B, H, S, head_dim), keys: (B, H, L, head_dim) -> (B, H, S, L)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))  # (B, H, S, L)
        attn_scores = attn_scores * self.pool_scale

        # Step 8: Mask out positions not in segment
        # segment_mask is (B, L, S), we need (B, 1, S, L) for broadcasting across heads
        segment_mask_transposed = segment_mask.transpose(1, 2).unsqueeze(1)  # (B, 1, S, L)
        attn_scores = attn_scores.masked_fill(segment_mask_transposed == 0, float('-inf'))

        # Step 9: Compute attention weights per segment
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, S, L)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Step 10: Apply attention: (B, H, S, L) @ (B, H, L, head_dim) -> (B, H, S, head_dim)
        pooled = torch.matmul(attn_weights, values)  # (B, H, S, head_dim)

        # Step 11: Concatenate heads back together
        pooled = pooled.transpose(1, 2).contiguous()  # (B, S, H, head_dim)
        pooled = pooled.view(batch_size, max_segments, hidden_dim)  # (B, S, D)

        # Step 12: Output projection to combine information from all heads
        pooled = self.pool_output(pooled)

        # Ensure output maintains same dtype as input
        pooled = pooled.to(dtype=dtype)

        return pooled  # B x S x D

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
        batch_size = hidden.size(0)
        seq_len = hidden.size(1)
        device = hidden.device

        # Step 1: UNet processes full hidden states for segmentation
        unet_output = self.unet(hidden)  # (B, L, 768)

        # Add residual connection
        enhanced_hidden = hidden + unet_output  # (B, L, 768)

        # Step 2: Apply BP2 Q/K logic on enhanced features
        # Q path: process tokens [0:L-1]
        q_hidden = self.q_proj_layer(F.normalize(
            self.q_mlp(F.normalize(self.dropout(enhanced_hidden[:, :-1]), dim=-1)),
            dim=-1))

        # K path: process tokens [1:L]
        k_hidden = self.k_proj_layer(F.normalize(
            self.k_mlp(F.normalize(self.dropout(enhanced_hidden[:, 1:]), dim=-1)),
            dim=-1))

        # Step 3: Cosine similarity for boundary detection
        cos_sim = torch.einsum("bld,bld->bl", q_hidden, k_hidden)  # (B, L-1)

        # Convert similarity to boundary probability
        probs = torch.clamp((1 - (cos_sim + self.similarity_bias)) * 0.5, 0, 1)  # (B, L-1)
        probs = F.pad(probs, (0, 1), "constant", 0.0)  # (B, L)

        # Step 4: Boundary sampling
        if self.training:
            # Use RelaxedBernoulli for differentiable sampling
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp,
                probs=probs,
            )
            soft_boundaries = bernoulli.rsample()
            hard_samples = (soft_boundaries > self.threshold).float()
        else:
            # During evaluation, use probabilities directly
            soft_boundaries = probs
            hard_samples = (probs > self.threshold).float()

        # Apply attention mask if provided
        if attention_mask is not None:
            soft_boundaries = soft_boundaries * attention_mask
            hard_samples = hard_samples * attention_mask

            # Ensure the last real token is always a boundary
            pad_mask = attention_mask == 0
            if pad_mask.any():
                first_pad_mask = pad_mask & (pad_mask.long().cumsum(dim=1) == 1)
                last_real_mask = torch.roll(first_pad_mask, shifts=-1, dims=1)
                last_real_mask[:, -1] = False
                last_real_mask = last_real_mask.float()
                hard_samples = torch.maximum(hard_samples, last_real_mask)
                soft_boundaries = torch.maximum(soft_boundaries, last_real_mask)

        # Straight-through estimator for gradient flow
        hard_boundaries = hard_samples - soft_boundaries.detach() + soft_boundaries

        # Step 5: Pooling
        if self.use_attention_pooling:
            pooled = self._attention_pooling(hard_boundaries, enhanced_hidden, attention_mask)  # B x S x D
        else:
            pooled = self._weighted_mean_pooling(hard_boundaries, enhanced_hidden, attention_mask)  # B x S x D

        pooled = self._add_positional_embeddings(pooled)

        # Create shortened attention mask
        shortened_attention_mask = None
        if attention_mask is not None:
            keep_mask = hard_boundaries == 1
            shortened_masks = []

            for b in range(batch_size):
                keep_indices = keep_mask[b].nonzero(as_tuple=True)[0]
                original_mask = attention_mask[b]
                shortened_mask = original_mask[keep_indices]
                shortened_masks.append(shortened_mask)

            shortened_attention_mask = pad_sequence(
                shortened_masks, batch_first=True, padding_value=0.0
            )

        # Calculate boundary statistics
        num_boundaries_tensor = hard_boundaries.sum()
        if attention_mask is not None:
            total_positions_tensor = attention_mask.sum()
        else:
            total_positions_tensor = torch.tensor(
                hard_boundaries.numel(),
                device=hard_boundaries.device,
                dtype=torch.float,
            )

        # Loss computation
        if self.training:
            if target_boundary_counts is not None:
                per_sample_loss = 10. * self.calc_loss_target_counts(
                    hard_boundaries,
                    attention_mask,
                    target_boundary_counts,
                    reduce=False,
                )
            else:
                per_sample_loss = 10 * self.calc_example_loss(
                    hard_boundaries,
                    attention_mask,
                    reduce=False
                )
            loss = per_sample_loss if return_unreduced_boundary_loss else per_sample_loss.mean()
        else:
            # Don't calculate loss during evaluation
            loss = torch.tensor(0.0, device=hidden.device)
            if return_unreduced_boundary_loss:
                loss = loss.repeat(batch_size)

        num_boundaries = num_boundaries_tensor.item()
        total_positions = total_positions_tensor.item()

        # Log probabilities for RL
        log_prob = None
        if return_log_probs:
            probs_clamped = torch.clamp(probs, min=1e-8, max=1 - 1e-8)
            log_probs_map = torch.where(
                hard_samples == 1,
                torch.log(probs_clamped),
                torch.log1p(-probs_clamped)
            )
            if attention_mask is not None:
                log_probs_map = log_probs_map * attention_mask
            log_prob = log_probs_map.sum(dim=1)

        # Confidence scores
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

        # Entropy
        entropy = None
        if return_entropy:
            probs_clamped = torch.clamp(probs, min=1e-8, max=1 - 1e-8).to(torch.float32)
            entropy_map = -(
                probs_clamped * torch.log(probs_clamped)
                + (1.0 - probs_clamped) * torch.log1p(-probs_clamped)
            )
            if attention_mask is not None:
                entropy_map = entropy_map * attention_mask.to(entropy_map.dtype)
            entropy = entropy_map.sum(dim=1)

        # Calculate boundary spacing statistics
        boundary_cv = None
        boundary_adjacent_pct = None

        with torch.no_grad():
            all_spacings = []
            adjacent_count = 0
            total_boundaries = 0

            for b in range(batch_size):
                # Get boundary positions for this sample
                if attention_mask is not None:
                    valid_length = int(attention_mask[b].sum().item())
                    boundaries_b = hard_samples[b, :valid_length]
                else:
                    boundaries_b = hard_samples[b]

                # Find positions where boundaries occur
                boundary_positions = boundaries_b.nonzero(as_tuple=True)[0]

                if len(boundary_positions) > 1:
                    # Calculate spacings between consecutive boundaries
                    spacings = boundary_positions[1:] - boundary_positions[:-1]
                    all_spacings.extend(spacings.cpu().tolist())

                    # Count adjacent boundaries (spacing == 1)
                    adjacent_count += (spacings == 1).sum().item()
                    total_boundaries += len(boundary_positions) - 1  # Number of gaps between boundaries

            # Calculate coefficient of variation (CV = std / mean)
            if len(all_spacings) > 0:
                spacings_tensor = torch.tensor(all_spacings, dtype=torch.float32)
                mean_spacing = spacings_tensor.mean()
                std_spacing = spacings_tensor.std()
                if mean_spacing > 0:
                    boundary_cv = (std_spacing / mean_spacing).item()
                else:
                    boundary_cv = 0.0

            # Calculate adjacent percentage
            if total_boundaries > 0:
                boundary_adjacent_pct = (adjacent_count / total_boundaries) * 100.0
            else:
                boundary_adjacent_pct = 0.0

        return (
            pooled,
            loss,
            num_boundaries,
            total_positions,
            shortened_attention_mask,
            log_prob,
            confidence,
            entropy,
            boundary_cv,
            boundary_adjacent_pct,
        )

    def calc_loss(self, num_boundaries, total_positions):
        scheduled_prior = self.get_scheduled_prior()
        loss = binomial_loss(num_boundaries, total_positions, scheduled_prior)
        return loss * self.boundary_loss_weight

    def calc_example_loss(self, hard_boundaries, attention_mask, reduce=True):
        """Calculates binomial loss per example using the scheduled prior."""
        per_item_boundaries = hard_boundaries.sum(dim=1)
        if attention_mask is not None:
            per_item_totals = attention_mask.sum(dim=1)
        else:
            per_item_totals = torch.full_like(
                per_item_boundaries, hard_boundaries.size(1), dtype=torch.float)

        scheduled_prior = self.get_scheduled_prior()

        device = per_item_boundaries.device
        prior_tensor = torch.tensor(
            scheduled_prior, device=device, dtype=torch.float32)

        total_positions_tensor = per_item_totals.to(
            device=device, dtype=torch.float32)
        num_boundaries_tensor = per_item_boundaries.to(
            device=device, dtype=torch.float32)

        binomial_dist = torch.distributions.binomial.Binomial(
            total_positions_tensor,
            probs=prior_tensor
        )

        loss_values = -binomial_dist.log_prob(num_boundaries_tensor)
        normalized_loss = loss_values / total_positions_tensor.clamp(min=1.0)

        per_example_loss = normalized_loss * self.boundary_loss_weight

        if reduce:
            return per_example_loss.mean()
        return per_example_loss

    def calc_loss_target_counts_flexible(
        self,
        boundaries,
        attention_mask,
        target_boundary_counts,
        reduce=True,
    ):
        """
        Calculate loss using either soft or hard boundaries.

        Args:
            boundaries: Either soft boundaries (continuous 0-1) or hard boundaries (discrete 0/1)
            attention_mask: Mask for valid positions
            target_boundary_counts: Target number of boundaries per sequence
            reduce: Whether to reduce the loss to a scalar

        Returns:
            Loss tensor (scalar if reduce=True, per-sample if reduce=False)
        """
        loss_values = binomial_loss_from_target_counts_flexible(
            boundaries,
            attention_mask,
            target_boundary_counts,
        )

        if reduce:
            return loss_values.mean()
        return loss_values

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
