import torch


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
    prior_tensor = torch.tensor(prior, device=device)

    # Create binomial distribution with total_positions trials and prior probability
    binomial = torch.distributions.binomial.Binomial(
        total_positions,
        probs=prior_tensor
    )

    # Calculate negative log likelihood of observing num_boundaries
    loss = -binomial.log_prob(num_boundaries).mean() / total_positions

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
