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
