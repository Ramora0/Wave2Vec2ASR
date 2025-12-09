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
        entropy_bonus_weight=0.001,
        whisper_learning_rate=None,
    ):
        """
        Args:
            model: MagnetWhisper model with boundary predictors
            num_samples: Number of boundary samples per audio (K in GRPO)
            clip_eps: PPO clipping epsilon
            learning_rate: Learning rate for boundary predictor
            normalize_advantages_flag: Whether to normalize advantages
            freeze_non_boundary: If True, only train boundary predictors (ignored if whisper_learning_rate is set)
            base_batch_size: Nominal dataloader batch size (before K expansion)
            callbacks: Optional iterable of callables invoked on training events
            amp_enabled: Enable autocast mixed precision (CUDA only)
            amp_dtype: Autocast dtype to use when amp_enabled
            entropy_bonus_weight: Weight for entropy bonus in loss (default: 0.01)
            whisper_learning_rate: Learning rate for Whisper model parameters (if None, only train boundary predictors)
        """
        self.model = model
        self.K = num_samples
        self.base_batch_size = base_batch_size
        self.clip_eps = clip_eps
        self.normalize_advantages_flag = normalize_advantages_flag
        self.amp_enabled = bool(amp_enabled) and torch.cuda.is_available()
        self.amp_dtype = amp_dtype if self.amp_enabled else None
        self.entropy_bonus_weight = entropy_bonus_weight

        # Determine if we should train the full model or just boundary predictors
        train_full_model = whisper_learning_rate is not None

        # Freeze model except boundary predictors if not training full model
        if not train_full_model and freeze_non_boundary:
            self._freeze_non_boundary_params()

        # Create optimizer with parameter groups for differential learning rates
        boundary_params = []
        whisper_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if "boundary_predictor" in name:
                boundary_params.append(param)
            else:
                whisper_params.append(param)

        if not boundary_params:
            raise ValueError(
                "No trainable boundary predictor parameters found! "
                "Make sure boundary predictor is initialized."
            )

        # Build parameter groups for optimizer
        param_groups = [
            {"params": boundary_params, "lr": learning_rate, "name": "boundary_predictor"}
        ]

        if train_full_model and whisper_params:
            param_groups.append(
                {"params": whisper_params, "lr": whisper_learning_rate, "name": "whisper_model"}
            )
            print(f"Training full model: {len(boundary_params)} boundary params (LR={learning_rate}), "
                  f"{len(whisper_params)} Whisper params (LR={whisper_learning_rate})")
        else:
            print(f"Training boundary predictors only: {len(boundary_params)} params (LR={learning_rate})")

        self.optimizer = AdamW(param_groups)
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
        """Freeze all parameters except boundary predictor."""
        for name, param in self.model.named_parameters():
            if "boundary_predictor" not in name:
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
        input_features_repeated = input_features.repeat(
            self.K, 1, 1)  # (K*B, C, T)
        labels_repeated = labels.repeat(self.K, 1)  # (K*B, L)
        attention_mask_repeated = attention_mask.repeat(
            self.K, 1) if attention_mask is not None else None
        target_boundary_counts_repeated = target_boundary_counts.repeat(
            1, self.K) if target_boundary_counts is not None else None

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
                boundary_rl=True,  # Enable RL mode for boundary predictors
                return_boundary_confidence=False,  # Get confidence estimates for diagnostics
                return_entropy=False,  # Get entropy for entropy bonus
            )

        # Extract log probabilities from model (summed across all boundary predictor layers)
        log_probs_all = self.model._boundary_log_probs  # (K*B,)
        if log_probs_all is None:
            raise RuntimeError(
                "Boundary log probabilities not returned; ensure return_boundary_log_probs=True")
        log_probs_all = log_probs_all.to(torch.float32)

        # ---- START DEBUGGING (fp16 overflow) ----
        if torch.isinf(log_probs_all).any() or torch.isnan(log_probs_all).any():
            print(
                "!!! Corrupted values detected in log_probs_all immediately after model forward pass !!!")
            print(f"Contains inf: {torch.isinf(log_probs_all).any()}")
            print(f"Contains NaN: {torch.isnan(log_probs_all).any()}")
            # Optional: Stop execution to inspect
            raise RuntimeError(
                "Overflow detected in model forward pass, resulting in inf/nan log_probs.")
        # ---- END DEBUGGING ----

        # Extract boundary loss for logging (move to CPU immediately)
        boundary_loss_all = self.model._boundary_loss
        if boundary_loss_all is None:
            boundary_loss_all = torch.zeros_like(outputs.loss)
        elif boundary_loss_all.dim() == 0:
            boundary_loss_all = boundary_loss_all.expand_as(outputs.loss)
        boundary_loss_all = boundary_loss_all.to(
            torch.float32).cpu()  # Move to CPU for logging

        boundary_conf_all = getattr(self.model, "_boundary_confidence", None)
        if boundary_conf_all is not None:
            # Already moved to CPU in MagnetWhisper
            boundary_conf_all = boundary_conf_all.to(torch.float32)

        # Extract entropy for bonus (already on CPU)
        entropy_all = getattr(self.model, "_entropy", None)
        if entropy_all is not None:
            entropy_all = entropy_all.to(torch.float32)

        # Reshape to (B, K) - each audio has K configurations
        # Note: per_sample_losses now contains TOTAL loss (ASR + boundary) for each sample
        # Move losses to CPU immediately as they're only used for rewards (detached)
        per_sample_losses = outputs.loss.to(torch.float32).cpu().view(
            self.K, batch_size).T  # (K, B) -> (B, K)
        # (K, B) -> (B, K) - KEEP ON GPU for gradients
        log_probs = log_probs_all.view(self.K, batch_size).T
        boundary_losses = boundary_loss_all.view(
            self.K, batch_size).T  # (K, B) -> (B, K) - already on CPU

        avg_confidence = None
        conf_std = None
        if boundary_conf_all is not None:
            try:
                conf_view = boundary_conf_all.view(
                    self.K, batch_size).T  # (B, K)
                avg_confidence = conf_view.mean().item()
                conf_std = conf_view.std().item()
            except RuntimeError:
                avg_confidence = None
                conf_std = None

        avg_entropy = None
        if entropy_all is not None:
            try:
                entropy_view = entropy_all.view(self.K, batch_size).T  # (B, K)
                avg_entropy = entropy_view.mean().item()
            except RuntimeError:
                avg_entropy = None

        # Extract boundary CV for diagnostics (already a scalar)
        boundary_cv = getattr(self.model, "_boundary_cv", None)

        # Get compression ratio
        compression_ratio = self.model.get_and_reset_compression_ratio()

        # === Step 2: Compute advantages from detached losses ===
        # Rewards are negative losses (detached - no gradients through ASR!)
        # Note: per_sample_losses is already on CPU, so advantages computed on CPU
        with torch.no_grad():
            # (B, K) - already detached and on CPU
            rewards_per_audio = -per_sample_losses

            # NOTE: Entropy bonus is removed from reward and applied to loss directly

            # Compute advantages for each audio independently (each gets its own baseline)
            advantages = torch.stack([
                # Compute for each audio's K samples
                compute_grpo_advantages(rewards_per_audio[b])
                for b in range(batch_size)
            ])  # (B, K) - on CPU

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
        # Move advantages to GPU only for this computation (log_probs is on GPU)
        advantages_gpu = advantages.to(log_probs.device)
        policy_loss = -(advantages_gpu * log_probs).mean()

        # Add entropy bonus directly to the loss to encourage exploration
        # if avg_entropy is not None and self.entropy_bonus_weight > 0:
        #     # We want to MAXIMIZE entropy, and the optimizer MINIMIZES the loss,
        #     # so we SUBTRACT the entropy bonus from the loss.
        #     entropy_loss = self.entropy_bonus_weight * avg_entropy
        #     policy_loss = policy_loss - entropy_loss

        total_policy_loss = policy_loss.item()

        # Backpropagate - gradients ONLY through log_probs → boundary predictors
        # NO gradients through downsample → encoder → decoder!
        if self.scaler.is_enabled():
            self.scaler.scale(policy_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            ).item()
            for name, param in self.model.named_parameters():
                if "boundary_predictors" in name and param.grad is not None:
                    grad_norm = param.grad.norm()
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"🚨 CORRUPTED GRADIENT: {name}")
                        print(
                            f"   grad min/max/norm: {param.grad.min()}/{param.grad.max()}/{grad_norm}")
                        print(
                            f"   param min/max: {param.data.min()}/{param.data.max()}")
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            ).item()
            self.optimizer.step()

        # Clear model's stored tensors to free GPU memory
        self.model._boundary_log_probs = None
        self.model._boundary_loss = None
        self.model._boundary_confidence = None
        self.model._entropy = None
        self.model._boundary_cv = None

        # === Step 4: Log metrics ===
        with torch.no_grad():
            # Separate ASR and boundary loss for more accurate logging
            pure_asr_loss = per_sample_losses - boundary_losses

            metrics = {
                "train/grpo_reward_mean": rewards_per_audio.mean().item(),
                "train/grpo_reward_std": rewards_per_audio.std().item(),
                "train/asr_loss": pure_asr_loss.mean().item(),
                "train/boundary_loss": boundary_losses.mean().item(),
                "train/total_loss": per_sample_losses.mean().item(),
                "train/compression_ratio": compression_ratio,
                "train/policy_loss": total_policy_loss,
                "train/grad_norm": grad_norm,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            }

            # Log learning rates for all parameter groups
            if len(self.optimizer.param_groups) > 1:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    group_name = param_group.get("name", f"group_{i}")
                    metrics[f"train/lr_{group_name}"] = param_group["lr"]

            if avg_confidence is not None:
                metrics["train/boundary_confidence"] = avg_confidence
                if conf_std is not None:
                    metrics["train/boundary_confidence_std"] = conf_std

            if avg_entropy is not None:
                metrics["train/entropy"] = avg_entropy

            if boundary_cv is not None:
                metrics["train/boundary_cv"] = boundary_cv

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
        total_steps = len(dataloader) if hasattr(
            dataloader, "__len__") else None
        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Train step
            metrics = self.train_step(batch)

            # Accumulate metrics
            epoch_metrics["reward"].append(metrics["train/grpo_reward_mean"])
            epoch_metrics["asr_loss"].append(metrics["train/asr_loss"])
            epoch_metrics["boundary_loss"].append(
                metrics["train/boundary_loss"])
            epoch_metrics["compression_ratio"].append(
                metrics["train/compression_ratio"])

            # Update progress bar with smoothed metrics
            comp_ratio = metrics.get("train/compression_ratio", 0.0)
            compression_display = float(
                'inf') if comp_ratio == 0 else 1.0 / comp_ratio
            pbar.set_postfix({
                "asr": f"{metrics['train/asr_loss']:.3f}",
                "boundary": f"{metrics['train/boundary_loss']:.3f}",
                "comp": f"{compression_display:.3f}",
            })

            # Log to wandb
            if self.global_step == 1 or (step + 1) % log_interval == 0:
                log_payload = dict(metrics)
                # Log epoch as a monotonically increasing value (epoch + progress)
                if total_steps and total_steps > 0:
                    epoch_progress = min(1.0, (step + 1) / total_steps)
                    # epoch is 1-indexed, so subtract 1
                    log_payload["epoch"] = epoch + epoch_progress - 1
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
