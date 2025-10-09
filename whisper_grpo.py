"""
GRPO training script for MagnetWhisper boundary prediction.

This script loads a pre-trained MagnetWhisper model (trained with compression-only)
and fine-tunes the boundary predictors using GRPO to optimize for ASR performance.
"""

import os
from pathlib import Path
import torch
import wandb
from tqdm import tqdm

from MagnetWhisper import MagnetWhisper
from librispeech import (
    create_librispeech_data_module,
    build_map_to_pred,
    test_inference_speed,
)
from grpo_trainer import GRPOBoundaryTrainer
from evaluate import load

# ============= Configuration =============

"""
- #samples
- boundary temperature
"""

# Model checkpoint (pre-trained with compression-only)
MODEL_CHECKPOINT = "./models/old-save"

# GRPO hyperparameters
BATCH_SIZE = 32  # Tuned for A100 (effective batch = 16 * 8 = 128 rollouts)
NUM_SAMPLES = 8  # Number of boundary samples per audio (K) - tuned for A100
CLIP_EPS = 0.2  # PPO clipping epsilon
LEARNING_RATE = 3e-6  # Lower LR for RL fine-tuning
NORMALIZE_ADVANTAGES = True  # Whether to normalize advantages

# Training settings
NUM_EPOCHS = 3
LOG_STEPS = 50 * 16 // BATCH_SIZE
EVAL_STEPS = 1500 * 16 // BATCH_SIZE
SAVE_STEPS = 2000 * 16 // BATCH_SIZE

# Model settings
BOUNDARY_TEMP = 2  # Temperature for boundary sampling
USE_FP16 = True
AMP_DTYPE = torch.float16
AMP_ENABLED = USE_FP16 and torch.cuda.is_available()

# Paths
MODEL_NAME = "magnet-grpo"
MODEL_DIR = Path("./models") / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _set_boundary_temperature(magnet_model, temperature):
    """Set temperature for boundary predictors."""
    predictors = getattr(magnet_model.model.encoder, "boundary_predictors", [])
    for predictor in predictors:
        if hasattr(predictor, "temp"):
            predictor.temp = temperature
    magnet_model.boundary_temperature = temperature


def evaluate_model_batched(model, eval_dataset, processor, batch_size=128, num_samples=None):
    """
    Evaluate model on a dataset with batched inference.

    Args:
        model: MagnetWhisper model
        eval_dataset: Evaluation dataset
        processor: Whisper processor
        batch_size: Batch size for inference
        num_samples: Number of samples to evaluate (None = all)

    Returns:
        metrics: Dict with WER and other metrics
    """
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    model.eval()

    # Sample subset if requested
    if num_samples is not None:
        eval_subset = eval_dataset.select(
            range(min(num_samples, len(eval_dataset))))
    else:
        eval_subset = eval_dataset

    # Create dataloader
    def collate_fn(batch):
        input_features = torch.stack(
            [torch.tensor(item["input_features"]) for item in batch])
        attention_masks = torch.stack(
            [torch.tensor(item["attention_mask"]) for item in batch])
        labels = [item["labels"] for item in batch]
        return {
            "input_features": input_features,
            "attention_mask": attention_masks,
            "labels": labels,
        }

    dataloader = DataLoader(
        eval_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Keep simple for now
    )

    all_predictions = []
    all_references = []

    # Run batched inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_features = batch["input_features"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")

            # Generate predictions
            with torch.cuda.amp.autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                predicted_ids = model.generate(
                    input_features,
                    attention_mask=attention_mask,
                )

            # Decode predictions
            predictions = processor.batch_decode(
                predicted_ids, skip_special_tokens=True)

            # Decode references
            references = []
            for label_ids in batch["labels"]:
                # Convert to tensor and decode
                label_tensor = torch.tensor(label_ids)
                reference = processor.decode(
                    label_ids, skip_special_tokens=True)
                references.append(reference)

            # Normalize
            predictions = [processor.tokenizer._normalize(
                p) for p in predictions]
            references = [processor.tokenizer._normalize(
                r) for r in references]

            all_predictions.extend(predictions)
            all_references.extend(references)

    # Compute WER
    wer_metric = load("wer")
    wer = wer_metric.compute(
        references=all_references,
        predictions=all_predictions
    )

    # Get compression ratio
    compression_ratio = model.get_and_reset_compression_ratio()

    metrics = {
        "eval/wer": wer * 100,
        "eval/compression_ratio": compression_ratio,
        "eval/num_samples": len(all_predictions),
    }

    model.train()
    return metrics


def main():
    print("=" * 60)
    print("GRPO Training for MagnetWhisper Boundary Prediction")
    print("=" * 60)

    # Set wandb project via environment variable (same as whisper.py)
    os.environ["WANDB_PROJECT"] = "glimpse-rl"

    # Initialize wandb
    wandb.init(
        config={
            "model_checkpoint": MODEL_CHECKPOINT,
            "num_samples": NUM_SAMPLES,
            "clip_eps": CLIP_EPS,
            "learning_rate": LEARNING_RATE,
            "normalize_advantages": NORMALIZE_ADVANTAGES,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "boundary_temp": BOUNDARY_TEMP,
        }
    )

    # Load pre-trained model
    print(f"\nLoading model from {MODEL_CHECKPOINT}...")
    model = MagnetWhisper.from_pretrained(MODEL_CHECKPOINT)
    model.to("cuda")

    # Optional torch.compile for faster training (opt-in)
    if os.environ.get("USE_TORCH_COMPILE", "0") == "1":
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile()")
        except Exception as compile_exc:
            print(
                f"torch.compile failed, continuing without it: {compile_exc}")

    # Set boundary predictor temperature (target_progress always 1.0 for RL)
    _set_boundary_temperature(model, BOUNDARY_TEMP)
    model.set_boundary_target_progress(1.0)

    # Configure generation
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    print(
        f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    boundary_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "boundary_predictors" in n
    )
    print(f"Boundary predictor parameters: {boundary_params / 1e6:.2f}M")

    # Load data
    print("\nLoading LibriSpeech data...")
    data_module = create_librispeech_data_module(
        model.config.decoder_start_token_id
    )
    dataset = data_module.dataset
    processor = data_module.processor
    data_collator = data_module.data_collator

    # Create dataloaders
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Eval samples: {len(dataset['validation.clean'])}")

    # Initialize GRPO trainer
    print("\nInitializing GRPO trainer...")
    grpo_trainer = GRPOBoundaryTrainer(
        model=model,
        num_samples=NUM_SAMPLES,
        clip_eps=CLIP_EPS,
        learning_rate=LEARNING_RATE,
        normalize_advantages_flag=NORMALIZE_ADVANTAGES,
        freeze_non_boundary=True,  # Only train boundary predictors
        amp_enabled=AMP_ENABLED,
        amp_dtype=AMP_DTYPE,
    )

    print(f"GRPO trainer initialized with K={NUM_SAMPLES} samples per audio")

    # Tracking best validation performance across callbacks and epoch evals
    best_wer = float('inf')

    def log_evaluation(eval_metrics, epoch_idx, step_idx, model_ref):
        """Pretty-print and log evaluation metrics; save best checkpoints."""
        nonlocal best_wer

        print(f"  WER: {eval_metrics['eval/wer']:.2f}%")
        print(f"  Compression: {eval_metrics['eval/compression_ratio']:.3f}")
        print(f"  Samples: {eval_metrics['eval/num_samples']}")

        wandb.log({
            **eval_metrics,
            "epoch": epoch_idx,
        }, step=step_idx)

        if eval_metrics['eval/wer'] < best_wer:
            best_wer = eval_metrics['eval/wer']
            best_model_path = MODEL_DIR / "best"
            print(
                f"🎉 New best WER: {best_wer:.2f}% - Saving to {best_model_path}"
            )
            model_ref.save_pretrained(best_model_path)

    def step_callback(event, trainer, epoch, step, global_step, metrics):
        """Run periodic evaluation and checkpointing during training."""
        if event != "step_end":
            return

        # Periodic checkpoints
        if SAVE_STEPS > 0 and global_step % SAVE_STEPS == 0:
            checkpoint_path = MODEL_DIR / f"checkpoint-step-{global_step}"
            print(f"\nSaving checkpoint to {checkpoint_path}")
            trainer.model.save_pretrained(checkpoint_path)

        # Periodic validation WER tracking
        if EVAL_STEPS > 0 and global_step % EVAL_STEPS == 0:
            print(
                f"\nEvaluating at global step {global_step} (epoch {epoch})..."
            )
            eval_metrics = evaluate_model_batched(
                trainer.model,
                dataset["validation.clean"],
                processor,
                num_samples=None,
            )
            log_evaluation(eval_metrics, epoch_idx=epoch,
                           step_idx=global_step, model_ref=trainer.model)

    grpo_trainer.register_callback(step_callback)

    # Training loop driven by the trainer (callbacks handle eval/save)
    print("\n" + "=" * 60)
    print("Starting GRPO Training")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        avg_metrics = grpo_trainer.train_epoch(
            train_dataloader,
            epoch=epoch + 1,
            log_interval=LOG_STEPS,
        )

        print("\nEpoch Summary:")
        print(f"  Average Reward: {avg_metrics['reward']:.3f}")
        print(f"  Average ASR Loss: {avg_metrics['asr_loss']:.3f}")
        print(f"  Average Compression: {avg_metrics['compression_ratio']:.3f}")

        wandb.log(
            {
                "train/epoch_reward": avg_metrics["reward"],
                "train/epoch_asr_loss": avg_metrics["asr_loss"],
                "train/epoch_compression": avg_metrics["compression_ratio"],
                "train/epoch_progress": (epoch + 1) / NUM_EPOCHS,
                "epoch": epoch + 1,
            },
            step=grpo_trainer.global_step,
        )

        # Evaluate on the full validation set at the end of each epoch
        print(f"\nEvaluating after epoch {epoch + 1}...")
        eval_metrics = evaluate_model_batched(
            model, dataset["validation.clean"], processor, num_samples=None
        )
        log_evaluation(
            eval_metrics,
            epoch_idx=epoch + 1,
            step_idx=grpo_trainer.global_step,
            model_ref=model,
        )

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    # Load best model
    best_model_path = MODEL_DIR / "best"
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}...")
        model = MagnetWhisper.from_pretrained(best_model_path)
        model.to("cuda")

    # Test set evaluation (batched, full test set)
    print("\nEvaluating on full test set...")
    test_metrics = evaluate_model_batched(
        model,
        dataset["test.clean"],
        processor,
        num_samples=None
    )
    # Convert back to ratio for display
    test_wer = test_metrics["eval/wer"] / 100

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test WER: {test_wer * 100:.2f}%")
    print(f"Test samples: {test_metrics['eval/num_samples']}")
    print(f"Best validation WER: {best_wer:.2f}%")

    wandb.log({
        "final/test_wer": test_wer * 100,
        "final/test_samples": test_metrics['eval/num_samples'],
    })

    wandb.finish()
    print("\nTraining complete! 🎉")


if __name__ == "__main__":
    main()
