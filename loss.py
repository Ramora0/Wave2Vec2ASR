import torch


def _to_device_tensor(value, device, dtype=torch.float32):
    """Utility to move scalars or tensors onto the target device."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def binomial_loss(num_boundaries, total_positions, prior):
    """
    Calculate binomial loss from boundary counts.

    Args:
        num_boundaries: Number of boundaries (tensor)
        total_positions: Total number of positions (tensor)  
        prior: Prior probability (float)
    """
    # Get device for tensor creation
    device = num_boundaries.device if hasattr(
        num_boundaries, 'device') else torch.device('cpu')
    prior_tensor = _to_device_tensor(prior, device)
    total_positions_tensor = _to_device_tensor(total_positions, device)
    num_boundaries_tensor = _to_device_tensor(num_boundaries, device)

    # Create binomial distribution with total_positions trials and prior probability
    binomial = torch.distributions.binomial.Binomial(
        total_positions_tensor,
        probs=prior_tensor
    )

    # Calculate negative log likelihood of observing num_boundaries
    loss = -binomial.log_prob(num_boundaries_tensor)
    normalized_loss = loss / total_positions_tensor.clamp(min=1.0)

    return normalized_loss.mean()


def binomial_loss_from_target_counts(num_boundaries, total_positions, target_counts, eps=1e-6):
    """Binomial loss where the expected count matches target boundary counts."""
    if not isinstance(num_boundaries, torch.Tensor):
        raise ValueError(
            "num_boundaries must be a tensor for per-example loss computation")

    device = num_boundaries.device
    num_boundaries = num_boundaries.to(dtype=torch.float32)
    total_positions = _to_device_tensor(total_positions, device)
    target_counts = _to_device_tensor(target_counts, device)

    clamped_totals = total_positions.clamp(min=1.0)
    clamped_targets = torch.minimum(target_counts, clamped_totals)
    target_probs = (clamped_targets /
                    clamped_totals).clamp(min=eps, max=1 - eps)

    binomial = torch.distributions.binomial.Binomial(
        total_count=clamped_totals,
        probs=target_probs
    )
    loss = -binomial.log_prob(num_boundaries)
    return loss / clamped_totals


def binomial_loss_from_target_counts_flexible(
    boundaries,
    attention_mask,
    target_counts,
    eps=1e-6
):
    """
    Binomial loss that works with both soft and hard boundaries.

    Args:
        boundaries: Either soft boundaries (continuous values 0-1) or hard boundaries (discrete 0/1)
                   Shape: (batch_size, seq_len)
        attention_mask: Mask indicating valid positions. Shape: (batch_size, seq_len)
        target_counts: Target number of boundaries per sequence. Shape: (batch_size,)
        eps: Small value to prevent numerical issues

    Returns:
        torch.Tensor: Per-sample loss values. Shape: (batch_size,)
    """
    if not isinstance(boundaries, torch.Tensor):
        raise ValueError("boundaries must be a tensor")

    device = boundaries.device
    boundaries = boundaries.to(dtype=torch.float32)
    target_counts = _to_device_tensor(
        target_counts, device, dtype=torch.float32)

    # Calculate expected number of boundaries per sample
    if attention_mask is not None:
        attention_mask = attention_mask.to(dtype=torch.float32)
        # Sum boundaries only over valid positions
        expected_boundaries = (boundaries * attention_mask).sum(dim=1)
        # Count total valid positions per sample
        total_positions = attention_mask.sum(dim=1)
    else:
        # No masking - use all positions
        expected_boundaries = boundaries.sum(dim=1)
        total_positions = torch.full(
            (boundaries.size(0),),
            boundaries.size(1),
            device=device,
            dtype=torch.float32
        )

    # Clamp to avoid numerical issues
    clamped_totals = total_positions.clamp(min=1.0)
    clamped_targets = torch.minimum(target_counts, clamped_totals)
    target_probs = (clamped_targets /
                    clamped_totals).clamp(min=eps, max=1 - eps)

    # For soft boundaries, we compute the expected negative log-likelihood
    # under the binomial distribution using the continuous relaxation

    # The binomial log-likelihood for count k out of n trials with probability p is:
    # log P(k|n,p) = log(n choose k) + k*log(p) + (n-k)*log(1-p)
    #
    # For soft boundaries, we replace the discrete count k with the expected count
    # and compute the expected log-likelihood. The binomial coefficient term is constant
    # for a given n and target, so we can ignore it for optimization purposes.

    # Expected log-likelihood terms:
    # E[k * log(p)] = expected_boundaries * log(target_probs)
    # E[(n-k) * log(1-p)] = (total_positions - expected_boundaries) * log(1 - target_probs)

    log_target_probs = torch.log(target_probs.clamp(min=eps))
    log_one_minus_target_probs = torch.log((1 - target_probs).clamp(min=eps))

    # Expected negative log-likelihood (we want to minimize this)
    loss = -(
        expected_boundaries * log_target_probs +
        (clamped_totals - expected_boundaries) * log_one_minus_target_probs
    )

    # Normalize by total positions for scale consistency
    loss = loss / clamped_totals

    return loss


def hinge_loss(num_boundaries, total_positions, prior_mean, prior_std, s_bound=3.0):
    """
    Compute hinge loss for boundary predictions centered at prior_mean.

    Args:
        num_boundaries (torch.Tensor): Number of boundaries
        total_positions (torch.Tensor): Total number of positions
        prior_mean (float): Expected mean boundary probability (center of the bounds)
        prior_std (float): Standard deviation of boundary probability
        s_bound (float): Scaling factor for both bounds (default: 3.0)

    Returns:
        torch.Tensor: Scalar loss value
    """
    # Estimate actual boundary probability
    est_prior = num_boundaries / total_positions

    # Convert prior values to tensors on the same device
    device = num_boundaries.device if hasattr(
        num_boundaries, 'device') else torch.device('cpu')
    prior_mean_tensor = torch.tensor(prior_mean, device=device)
    prior_std_tensor = torch.tensor(prior_std, device=device)

    # Define bounds centered at prior_mean
    upper_bound = prior_mean_tensor + s_bound * prior_std_tensor
    lower_bound = prior_mean_tensor - s_bound * prior_std_tensor

    # Calculate hinge losses
    loss_high = torch.clamp(est_prior - upper_bound, min=0.0)
    loss_low = torch.clamp(lower_bound - est_prior, min=0.0)

    # Combine and return total loss
    total_loss = loss_high + loss_low

    return total_loss


def repulsion_loss(hard_boundaries, attention_mask=None, kernel_size=5):
    """
    Repulsion loss that penalizes nearby boundaries using convolution.

    Uses a convolution with kernel_size to detect boundaries that are close together.
    The loss is 0 if there are no nearby boundaries.

    Args:
        hard_boundaries: Binary boundary indicators. Shape: (batch_size, seq_len)
        attention_mask: Optional mask for valid positions. Shape: (batch_size, seq_len)
        kernel_size: Size of the convolution kernel to detect nearby boundaries (default: 5)

    Returns:
        torch.Tensor: Scalar loss value (0 if no nearby boundaries)
    """
    if not isinstance(hard_boundaries, torch.Tensor):
        raise ValueError("hard_boundaries must be a tensor")

    # Convert to float32 for conv1d
    boundaries = hard_boundaries.to(dtype=torch.float32)

    # Create a convolution kernel: all ones except the center
    # This detects if there are boundaries near the current position
    kernel = torch.ones(1, 1, kernel_size, device=boundaries.device, dtype=torch.float32)
    center = kernel_size // 2
    kernel[0, 0, center] = 0.0  # Exclude the center position itself

    # Add channel dimension for conv1d: (batch_size, 1, seq_len)
    boundaries_3d = boundaries.unsqueeze(1)

    # Apply convolution with padding to maintain sequence length
    padding = kernel_size // 2
    nearby_count = torch.nn.functional.conv1d(
        boundaries_3d, kernel, padding=padding
    ).squeeze(1)  # Shape: (batch_size, seq_len)

    # For each boundary position, check if there are nearby boundaries
    # Multiply by the boundary indicator to only consider positions that are boundaries
    nearby_violations = nearby_count * boundaries

    # Apply attention mask if provided
    if attention_mask is not None:
        attention_mask = attention_mask.to(dtype=torch.float32)
        nearby_violations = nearby_violations * attention_mask
        valid_positions = attention_mask.sum()
    else:
        valid_positions = boundaries.numel()

    # Sum all violations and normalize
    total_violations = nearby_violations.sum()

    # Normalize by number of valid positions to make loss scale-invariant
    # Clamp valid_positions to avoid division by zero
    loss = total_violations / valid_positions.clamp(min=1.0)

    return loss


def count_normalized_repulsion_loss(hard_boundaries, attention_mask=None, kernel_size=5):
    """
    Count-normalized repulsion loss that penalizes boundary clustering without affecting total count.

    This version normalizes by the number of boundaries rather than sequence length,
    making it measure "average clustering per boundary" instead of "total clustering per position".
    This makes it invariant to the total boundary count, so it won't fight against the
    binomial loss that constrains the count.

    Args:
        hard_boundaries: Binary boundary indicators. Shape: (batch_size, seq_len)
        attention_mask: Optional mask for valid positions. Shape: (batch_size, seq_len)
        kernel_size: Size of the convolution kernel to detect nearby boundaries (default: 5)

    Returns:
        torch.Tensor: Scalar loss measuring average neighbors per boundary (count-invariant)
    """
    if not isinstance(hard_boundaries, torch.Tensor):
        raise ValueError("hard_boundaries must be a tensor")

    # Convert to float32 for conv1d
    boundaries = hard_boundaries.to(dtype=torch.float32)

    # Create a convolution kernel: all ones except the center
    # This detects if there are boundaries near the current position
    kernel = torch.ones(1, 1, kernel_size, device=boundaries.device, dtype=torch.float32)
    center = kernel_size // 2
    kernel[0, 0, center] = 0.0  # Exclude the center position itself

    # Add channel dimension for conv1d: (batch_size, 1, seq_len)
    boundaries_3d = boundaries.unsqueeze(1)

    # Apply convolution with padding to maintain sequence length
    padding = kernel_size // 2
    nearby_count = torch.nn.functional.conv1d(
        boundaries_3d, kernel, padding=padding
    ).squeeze(1)  # Shape: (batch_size, seq_len)

    # For each boundary position, check if there are nearby boundaries
    # Multiply by the boundary indicator to only consider positions that are boundaries
    nearby_violations = nearby_count * boundaries

    # Apply attention mask if provided
    if attention_mask is not None:
        attention_mask = attention_mask.to(dtype=torch.float32)
        nearby_violations = nearby_violations * attention_mask
        num_boundaries = (boundaries * attention_mask).sum()
    else:
        num_boundaries = boundaries.sum()

    # Sum all violations and normalize
    total_violations = nearby_violations.sum()

    # KEY DIFFERENCE: Normalize by boundary count instead of position count
    # This measures "average neighbors per boundary" which is count-invariant
    # Whether you have 10 or 100 boundaries, this measures clustering tendency
    loss = total_violations / num_boundaries.clamp(min=1.0)

    return loss


def local_rate_uniformity_loss(
    hard_boundaries,
    attention_mask=None,
    target_boundary_counts=None,
    min_window_size=30,
    max_window_size=70,
    num_samples=8,
):
    """
    Penalize non-uniform boundary distribution by checking local boundary rates using binomial loss.

    Enforces roughly even compression across the sequence by sampling random windows
    with random sizes and measuring the binomial loss for each window. This prevents
    clustering without affecting the total boundary count.

    Args:
        hard_boundaries: Binary boundary indicators. Shape: (batch_size, seq_len)
        attention_mask: Optional mask for valid positions. Shape: (batch_size, seq_len)
        target_boundary_counts: Target boundaries per sequence. Shape: (batch_size,)
        min_window_size: Minimum window size for random sampling (default: 30)
        max_window_size: Maximum window size for random sampling (default: 70)
        num_samples: Number of random windows to sample per sequence (default: 8)

    Returns:
        torch.Tensor: Scalar loss measuring non-uniformity using binomial loss
    """
    if not isinstance(hard_boundaries, torch.Tensor):
        raise ValueError("hard_boundaries must be a tensor")

    batch_size, seq_len = hard_boundaries.shape
    device = hard_boundaries.device
    boundaries = hard_boundaries.to(dtype=torch.float32)

    if target_boundary_counts is None:
        # If no target provided, use actual count as target
        if attention_mask is not None:
            target_boundary_counts = (boundaries * attention_mask.to(dtype=torch.float32)).sum(dim=1)
        else:
            target_boundary_counts = boundaries.sum(dim=1)

    # Calculate expected boundary rate (boundaries per position)
    if attention_mask is not None:
        attention_mask = attention_mask.to(dtype=torch.float32)
        seq_lengths = attention_mask.sum(dim=1)  # Actual length per sequence
    else:
        seq_lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.float32)

    # Expected rate: target_count / seq_length
    target_rates = target_boundary_counts.to(device=device, dtype=torch.float32) / seq_lengths.clamp(min=1.0)

    # Filter out sequences that are too short for minimum window size
    valid_mask = seq_lengths >= min_window_size
    if not valid_mask.any():
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    # Only process valid sequences
    valid_boundaries = boundaries[valid_mask]
    valid_seq_lengths = seq_lengths[valid_mask]
    valid_target_rates = target_rates[valid_mask]
    valid_batch_size = valid_boundaries.shape[0]

    # Sample random window sizes for each sample
    # Shape: (valid_batch_size, num_samples)
    window_sizes = torch.randint(
        min_window_size,
        max_window_size + 1,
        (valid_batch_size, num_samples),
        device=device
    )

    # Sample random start positions
    # max_starts needs to account for each window's size
    max_starts = (valid_seq_lengths.unsqueeze(1) - window_sizes).clamp(min=0).long()

    # Generate random start positions
    uniform_samples = torch.rand(valid_batch_size, num_samples, device=device)
    starts = (uniform_samples * (max_starts + 1).float()).long()

    # Use maximum window size to create indices, then mask out invalid positions
    # Shape: (valid_batch_size, num_samples, max_window_size)
    window_offsets = torch.arange(max_window_size, device=device).unsqueeze(0).unsqueeze(0)
    window_indices = starts.unsqueeze(2) + window_offsets

    # Clamp indices to valid range
    window_indices = window_indices.clamp(0, seq_len - 1)

    # Create mask for valid positions within each window
    # Shape: (valid_batch_size, num_samples, max_window_size)
    position_mask = window_offsets < window_sizes.unsqueeze(2)

    # Extract windows using gather
    expanded_boundaries = valid_boundaries.unsqueeze(1).expand(-1, num_samples, -1)
    # Gather: (valid_batch_size, num_samples, max_window_size)
    windows = torch.gather(expanded_boundaries, 2, window_indices)

    # Apply position mask to zero out positions beyond each window's size
    windows = windows * position_mask.float()

    # Count boundaries in each window
    # Shape: (valid_batch_size, num_samples)
    window_counts = windows.sum(dim=2)

    # Calculate binomial loss for all windows at once
    # Each window has its own size and rate
    target_rates_expanded = valid_target_rates.unsqueeze(1).expand(-1, num_samples)
    target_rates_clamped = torch.clamp(target_rates_expanded, min=1e-6, max=1.0 - 1e-6)

    # Create binomial distribution with varying total_count per window
    binomial = torch.distributions.binomial.Binomial(
        total_count=window_sizes.float(),
        probs=target_rates_clamped
    )

    # Calculate loss for all windows
    # Shape: (valid_batch_size, num_samples)
    window_losses = -binomial.log_prob(window_counts) / window_sizes.float()

    # Average across all samples
    return window_losses.mean()
