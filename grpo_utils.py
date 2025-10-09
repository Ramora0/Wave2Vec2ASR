"""
GRPO utilities for reward computation and RL training helpers.
"""

import torch


def compute_grpo_reward(
    asr_loss,
    binomial_loss,
    compression_penalty_weight=0.1,
):
    """
    Compute GRPO reward combining ASR performance and compression constraint.

    Args:
        asr_loss: Cross-entropy loss from decoder (scalar)
        binomial_loss: Binomial loss from boundary predictor (scalar)
        compression_penalty_weight: Weight for binomial loss term

    Returns:
        reward: Scalar reward (higher is better)
    """
    # Negative because we want to maximize reward = minimize loss
    reward = -asr_loss - compression_penalty_weight * binomial_loss
    return reward


def compute_grpo_advantages(rewards, baseline=None):
    """
    Compute advantages for GRPO update.

    Args:
        rewards: (K,) tensor of rewards for K samples
        baseline: Optional baseline. If None, uses mean of rewards (group baseline)

    Returns:
        advantages: (K,) tensor of advantages
    """
    if baseline is None:
        baseline = rewards.mean()

    advantages = rewards - baseline
    return advantages


def normalize_advantages(advantages, eps=1e-8):
    """
    Normalize advantages to have zero mean and unit variance.
    This can help stabilize training.

    Args:
        advantages: (K,) tensor of advantages
        eps: Small constant for numerical stability

    Returns:
        normalized_advantages: (K,) tensor
    """
    mean = advantages.mean()
    std = advantages.std()
    normalized = (advantages - mean) / (std + eps)
    return normalized


def compute_policy_ratio(log_probs_new, log_probs_old):
    """
    Compute importance sampling ratio for policy gradient.

    Args:
        log_probs_new: Log probs under current policy
        log_probs_old: Log probs under old policy (from rollout)

    Returns:
        ratio: exp(log_probs_new - log_probs_old)
    """
    ratio = torch.exp(log_probs_new - log_probs_old)
    return ratio


def clipped_policy_loss(ratio, advantages, clip_eps=0.2):
    """
    Compute PPO-style clipped policy loss.

    Args:
        ratio: Importance sampling ratio π_new / π_old
        advantages: Advantage estimates
        clip_eps: Clipping epsilon (typically 0.2)

    Returns:
        loss: Scalar loss to minimize
    """
    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()
    return loss


def log_grpo_metrics(
    rewards,
    advantages,
    ratios,
    asr_losses,
    binomial_losses,
    compression_ratios,
    step,
    logger=None,
):
    """
    Log GRPO training metrics.

    Args:
        rewards: (K,) rewards
        advantages: (K,) advantages
        ratios: (K,) policy ratios
        asr_losses: (K,) ASR losses
        binomial_losses: (K,) binomial losses
        compression_ratios: (K,) compression ratios
        step: Training step
        logger: Optional wandb logger

    Returns:
        metrics: Dict of metrics
    """
    metrics = {
        "grpo/reward_mean": rewards.mean().item(),
        "grpo/reward_std": rewards.std().item(),
        "grpo/reward_max": rewards.max().item(),
        "grpo/reward_min": rewards.min().item(),
        "grpo/advantage_mean": advantages.mean().item(),
        "grpo/advantage_std": advantages.std().item(),
        "grpo/ratio_mean": ratios.mean().item(),
        "grpo/ratio_max": ratios.max().item(),
        "grpo/ratio_min": ratios.min().item(),
        "grpo/asr_loss_mean": torch.stack(asr_losses).mean().item() if isinstance(asr_losses, list) else asr_losses.mean().item(),
        "grpo/binomial_loss_mean": torch.stack(binomial_losses).mean().item() if isinstance(binomial_losses, list) else binomial_losses.mean().item(),
        "grpo/compression_ratio_mean": torch.stack(compression_ratios).mean().item() if isinstance(compression_ratios, list) else compression_ratios.mean().item(),
    }

    if logger is not None:
        logger.log(metrics, step=step)

    return metrics
