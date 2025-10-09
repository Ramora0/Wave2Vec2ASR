"""
GRPO (Group Relative Policy Optimization) trainer for boundary prediction.

This implements a minimal RL approach that doesn't require a value network.
Instead, it uses group-based baselines by sampling K boundary configurations
for each audio and using their mean reward as the baseline.
"""

import torch
from torch.optim import AdamW
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast, GradScaler

from grpo_utils import (
    compute_grpo_advantages,
    normalize_advantages,
)


class GRPOBoundaryTrainer:
    """
    GRPO trainer for optimizing boundary predictions with RL.

    This trainer samples multiple boundary configurations (group size K) for each
    audio sample, computes rewards based on ASR performance and compression constraints,
    and uses the group mean as a baseline for advantage computation.
    """

    def __init__(
        self,
        model,
        num_samples=8,
        clip_eps=0.2,
        learning_rate=1e-6,
        normalize_advantages_flag=True,
        freeze_non_boundary=True,
        base_batch_size=None,
        callbacks=None,
        amp_enabled=False,
        amp_dtype=torch.float16,
    ):
        """
        Args:
            model: MagnetWhisper model with boundary predictors
            num_samples: Number of boundary samples per audio (K in GRPO)
            clip_eps: PPO clipping epsilon
            learning_rate: Learning rate for boundary predictor
            normalize_advantages_flag: Whether to normalize advantages
            freeze_non_boundary: If True, only train boundary predictors
            base_batch_size: Nominal dataloader batch size (before K expansion)
            callbacks: Optional iterable of callables invoked on training events
            amp_enabled: Enable autocast mixed precision (CUDA only)
            amp_dtype: Autocast dtype to use when amp_enabled
        """
        self.model = model
        self.K = num_samples
        self.base_batch_size = base_batch_size
        self.clip_eps = clip_eps
        self.normalize_advantages_flag = normalize_advantages_flag
        self.amp_enabled = bool(amp_enabled) and torch.cuda.is_available()
        self.amp_dtype = amp_dtype if self.amp_enabled else None

        # Freeze model except boundary predictors if requested
        if freeze_non_boundary:
            self._freeze_non_boundary_params()

        # Create optimizer for boundary predictors only
        boundary_params = []
        for name, param in model.named_parameters():
            if "boundary_predictors" in name and param.requires_grad:
                boundary_params.append(param)

        if not boundary_params:
            raise ValueError(
                "No trainable boundary predictor parameters found! "
                "Make sure boundary predictors are initialized."
            )

        self.optimizer = AdamW(boundary_params, lr=learning_rate)
        self.global_step = 0

        self.callbacks = list(callbacks) if callbacks else []

        scaler_enabled = self.amp_enabled and self.amp_dtype == torch.float16
        self.scaler = GradScaler(enabled=scaler_enabled)

    def register_callback(self, callback):
        """Register a training callback executed after each step."""
        self.callbacks.append(callback)

    def _run_callbacks(self, event, **info):
        for callback in self.callbacks:
            try:
                callback(event=event, trainer=self, **info)
            except Exception as exc:
                print(f"[GRPOBoundaryTrainer] callback error: {exc}")

    def _freeze_non_boundary_params(self):
        """Freeze all parameters except boundary predictors."""
        for name, param in self.model.named_parameters():
            if "boundary_predictors" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def train_step(self, batch):
        """
        Perform one GRPO training step.

        Args:
            batch: Dict with keys:
                - input_features: (B, C, T) audio features
                - labels: (B, L) target token ids
                - attention_mask: (B, T) attention mask
                - target_boundary_counts: (B, num_layers) target compression

        Returns:
            metrics: Dict of training metrics
        """
        self.model.train()

        input_features = batch["input_features"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask", None)
        target_boundary_counts = batch.get("target_boundary_counts", None)

        batch_size = input_features.shape[0]

        # === Step 1: Collect K samples (rollouts) - BATCHED ===
        # Repeat inputs K times to sample K different boundary configurations in one pass
        # Each copy gets different random noise in rsample(), giving us K diverse samples
        input_features_repeated = input_features.repeat(self.K, 1, 1)  # (K*B, C, T)
        labels_repeated = labels.repeat(self.K, 1)  # (K*B, L)
        attention_mask_repeated = attention_mask.repeat(self.K, 1) if attention_mask is not None else None
        target_boundary_counts_repeated = target_boundary_counts.repeat(1, self.K) if target_boundary_counts is not None else None

        # === Step 1: Single forward pass for policy gradient ===
        # This computes BOTH losses (for rewards) AND log_probs (for gradients)
        # Losses are detached for reward signal, log_probs keep gradients
        with autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
            outputs = self.model(
                input_features=input_features_repeated,
                labels=labels_repeated,
                attention_mask=attention_mask_repeated,
                target_boundary_counts=target_boundary_counts_repeated,
                return_unreduced_loss=True,  # Get per-sample ASR losses (K*B,)
                return_boundary_log_probs=True,  # Get log probs for policy gradient (K*B,)
                return_unreduced_boundary_loss=True,  # Get per-sample boundary losses (K*B,)
                return_boundary_masks=True,  # Get hard boundary masks for diversity metrics
            )

        # Extract log probabilities from model (summed across all boundary predictor layers)
        log_probs_all = self.model._boundary_log_probs  # (K*B,)
        if log_probs_all is None:
            raise RuntimeError("Boundary log probabilities not returned; ensure return_boundary_log_probs=True")
        log_probs_all = log_probs_all.to(torch.float32)

        # Extract boundary loss for logging
        boundary_loss_all = self.model._boundary_loss
        if boundary_loss_all is None:
            boundary_loss_all = torch.zeros_like(outputs.loss)
        elif boundary_loss_all.dim() == 0:
            boundary_loss_all = boundary_loss_all.expand_as(outputs.loss)
        boundary_loss_all = boundary_loss_all.to(torch.float32)

        boundary_masks_all = getattr(self.model, "_boundary_masks", None)
        boundary_diversity = None
        if boundary_masks_all is not None:
            try:
                boundary_masks = boundary_masks_all.to(torch.float32).view(self.K, batch_size, -1).permute(1, 0, 2)
                if self.K > 1 and boundary_masks.size(-1) > 0:
                    cdf = boundary_masks.cumsum(dim=-1)
                    total_counts = cdf[:, :, -1:].clamp(min=1.0)
                    cdf = cdf / total_counts
                    diffs = []
                    for i in range(self.K):
                        for j in range(i + 1, self.K):
                            diffs.append((cdf[:, i] - cdf[:, j]).abs().mean(dim=1))
                    if diffs:
                        boundary_diversity = torch.stack(diffs, dim=0).mean().item()
            except RuntimeError:
                boundary_diversity = None

        # Reshape to (B, K) - each audio has K configurations
        # Note: per_sample_losses now contains TOTAL loss (ASR + boundary) for each sample
        per_sample_losses = outputs.loss.to(torch.float32).view(self.K, batch_size).T  # (K, B) -> (B, K)
        log_probs = log_probs_all.view(self.K, batch_size).T  # (K, B) -> (B, K)
        boundary_losses = boundary_loss_all.view(self.K, batch_size).T  # (K, B) -> (B, K)

        # Get compression ratio
        compression_ratio = self.model.get_and_reset_compression_ratio()

        # === Step 2: Compute advantages from detached losses ===
        # Rewards are negative losses (detached - no gradients through ASR!)
        with torch.no_grad():
            rewards_per_audio = -per_sample_losses.detach()  # (B, K)

            # Compute advantages for each audio independently (each gets its own baseline)
            advantages = torch.stack([
                compute_grpo_advantages(rewards_per_audio[b])  # Compute for each audio's K samples
                for b in range(batch_size)
            ])  # (B, K)

        if self.normalize_advantages_flag:
            advantages = normalize_advantages(advantages)

        # === Step 3: Policy Gradient Update ===
        # Use REINFORCE: maximize E[advantage * log_prob]
        # Gradients flow ONLY through log_probs, NOT through losses/rewards
        self.optimizer.zero_grad()

        # Policy gradient loss: -mean(advantage * log_prob)
        # Negative because we want to maximize, but optimizer minimizes
        # advantages are detached (pure reward signal, no gradients)
        # log_probs have gradients flowing back to boundary_mlp
        policy_loss = -(advantages.detach() * log_probs).mean()
        total_policy_loss = policy_loss.item()

        # Backpropagate - gradients ONLY through log_probs → boundary predictors
        # NO gradients through downsample → encoder → decoder!
        if self.scaler.is_enabled():
            self.scaler.scale(policy_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            ).item()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            ).item()
            self.optimizer.step()

        # === Step 4: Log metrics ===
        with torch.no_grad():
            metrics = {
                "train/grpo_reward_mean": rewards_per_audio.mean().item(),
                "train/grpo_reward_std": rewards_per_audio.std().item(),
                "train/asr_loss": per_sample_losses.mean().item(),
                "train/boundary_loss": boundary_losses.mean().item(),
                "train/compression_ratio": compression_ratio,
                "train/policy_loss": total_policy_loss,
                "train/grad_norm": grad_norm,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            }

            if boundary_diversity is not None:
                metrics["train/boundary_diversity"] = boundary_diversity

            # Add boundary predictor settings (similar to whisper.py)
        self.global_step += 1

        return metrics

    def train_epoch(self, dataloader, epoch, log_interval=10):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader yielding batches
            epoch: Current epoch number
            log_interval: Log metrics every N steps

        Returns:
            avg_metrics: Dict of average metrics for the epoch
        """
        epoch_metrics = {
            "reward": [],
            "asr_loss": [],
            "boundary_loss": [],
            "compression_ratio": [],
        }

        pbar = tqdm(dataloader, desc=f"GRPO Epoch {epoch}")
        total_steps = len(dataloader) if hasattr(dataloader, "__len__") else None
        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Train step
            metrics = self.train_step(batch)

            # Accumulate metrics
            epoch_metrics["reward"].append(metrics["train/grpo_reward_mean"])
            epoch_metrics["asr_loss"].append(metrics["train/asr_loss"])
            epoch_metrics["boundary_loss"].append(metrics["train/boundary_loss"])
            epoch_metrics["compression_ratio"].append(
                metrics["train/compression_ratio"])

            # Update progress bar with smoothed metrics
            comp_ratio = metrics.get("train/compression_ratio", 0.0)
            compression_display = float('inf') if comp_ratio == 0 else 1.0 / comp_ratio
            pbar.set_postfix({
                "asr": f"{metrics['train/asr_loss']:.3f}",
                "boundary": f"{metrics['train/boundary_loss']:.3f}",
                "comp": f"{compression_display:.3f}",
            })

            # Log to wandb
            if self.global_step == 1 or (step + 1) % log_interval == 0:
                log_payload = dict(metrics)
                if total_steps and total_steps > 0:
                    log_payload["train/epoch_progress"] = min(1.0, (step + 1) / total_steps)
                wandb.log(log_payload, step=self.global_step)

            # Invoke callbacks (e.g., periodic evaluation)
            self._run_callbacks(
                event="step_end",
                epoch=epoch,
                step=step,
                global_step=self.global_step,
                metrics=metrics,
            )

        # Compute average metrics
        avg_metrics = {
            k: sum(v) / len(v) if v else 0.0
            for k, v in epoch_metrics.items()
        }

        return avg_metrics
