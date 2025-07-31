import torch


def binomial_loss(num_boundaries, total_positions, prior, device):
    binomial = torch.distributions.binomial.Binomial(
        total_positions,
        probs=torch.Tensor([prior]).to(device)
    )
    loss_boundaries = -binomial.log_prob(num_boundaries).mean() / total_positions

    return loss_boundaries


def hinge_loss(preds, prior_mean, prior_std, s_bound=3.0):
    """
    Compute hinge loss for boundary predictions centered at prior_mean.

    Args:
        predictions (torch.Tensor): Predicted boundary probabilities, shape [batch_size, seq_len]
        prior_mean (float): Expected mean boundary probability (center of the bounds)
        prior_std (float): Standard deviation of boundary probability
        s_bound (float): Scaling factor for both bounds (default: 3.0)
        attention_mask (torch.Tensor, optional): Mask for padded tokens, shape [batch_size, seq_len]
                                               1 for real tokens, 0 for padding

    Returns:
        torch.Tensor: Scalar loss value
    """
    sum_preds = preds.sum(dim=-1)
    total_count = torch.full((preds.size(0),), preds.size(-1),
                             dtype=torch.float, device=preds.device)

    # Estimate actual boundary probability for each sequence
    est_prior = sum_preds / total_count

    # Convert prior values to tensors on the same device
    prior_mean_tensor = torch.tensor(prior_mean, device=preds.device)
    prior_std_tensor = torch.tensor(prior_std, device=preds.device)

    # Define bounds centered at prior_mean
    upper_bound = prior_mean_tensor + s_bound * prior_std_tensor
    lower_bound = prior_mean_tensor - s_bound * prior_std_tensor

    # Calculate hinge losses
    loss_high = torch.clamp(est_prior - upper_bound, min=0.0)
    loss_low = torch.clamp(lower_bound - est_prior, min=0.0)

    # Combine and return mean loss
    total_loss = (loss_high + loss_low).mean()

    return total_loss
