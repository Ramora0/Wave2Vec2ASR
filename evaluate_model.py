"""
Standalone evaluation script for comparing MagnetWhisper vs vanilla Whisper.

This script loads a model (either MagnetWhisper or vanilla Whisper),
evaluates it on LibriSpeech validation set, and reports WER.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from evaluate import load

from MagnetWhisper import MagnetWhisper
from transformers import WhisperForConditionalGeneration
from librispeech import create_librispeech_data_module


# ============= Configuration =============

# Model settings
USE_FP16 = True
AMP_DTYPE = torch.float16 if USE_FP16 else torch.float32
AMP_ENABLED = USE_FP16 and torch.cuda.is_available()

# Evaluation settings
BATCH_SIZE = 128  # Batch size for evaluation
EVAL_DATASET = "validation.clean"  # Options: "validation.clean", "test.clean"


def evaluate_model_batched(model, eval_dataset, processor, batch_size=128):
    """
    Evaluate model on a dataset with batched inference.

    Args:
        model: MagnetWhisper or Whisper model
        eval_dataset: Evaluation dataset
        processor: Whisper processor
        batch_size: Batch size for inference

    Returns:
        metrics: Dict with WER and other metrics
    """
    model.eval()

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
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    all_predictions = []
    all_references = []

    # Run batched inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_features = batch["input_features"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")

            # Generate predictions
            with torch.amp.autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE, device_type="cuda"):
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

    metrics = {
        "wer": wer * 100,
        "num_samples": len(all_predictions),
    }

    return metrics


def load_magnet_whisper(checkpoint_path=None, boundary_config=None):
    """
    Load a MagnetWhisper model.

    Args:
        checkpoint_path: Path to pretrained checkpoint (if None, loads base whisper-small)
        boundary_config: List of (layer, threshold) tuples for boundary predictors

    Returns:
        model: MagnetWhisper instance
    """
    if checkpoint_path:
        model = WhisperForConditionalGeneration.from_pretrained(
            checkpoint_path)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small",
            token="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"
        )

    # Convert to MagnetWhisper
    model.__class__ = MagnetWhisper

    # Load boundary predictors
    if boundary_config is None:
        boundary_config = []  # No boundary predictors by default

    model.load_magnet(boundary_config, "BoundaryPredictor1")

    return model


def load_vanilla_whisper(checkpoint_path=None):
    """
    Load a vanilla Whisper model.

    Args:
        checkpoint_path: Path to pretrained checkpoint (if None, loads base whisper-small)

    Returns:
        model: WhisperForConditionalGeneration instance
    """
    if checkpoint_path:
        model = WhisperForConditionalGeneration.from_pretrained(
            checkpoint_path)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small",
            token="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"
        )

    return model


def compare_encoder_hidden_states(vanilla_model, magnet_model, test_input, attention_mask):
    """
    Compare encoder hidden states between vanilla Whisper and MagnetWhisper.

    Args:
        vanilla_model: Vanilla Whisper model
        magnet_model: MagnetWhisper model (with no boundary predictors)
        test_input: Input features tensor
        attention_mask: Attention mask tensor

    Returns:
        (bool, torch.Tensor): Tuple of (are_identical, vanilla_hidden_state)
    """
    vanilla_model.eval()
    magnet_model.eval()

    with torch.no_grad():
        # Get encoder outputs from vanilla Whisper
        vanilla_encoder_outputs = vanilla_model.model.encoder(
            test_input,
            attention_mask=attention_mask,
            return_dict=True
        )
        vanilla_hidden = vanilla_encoder_outputs.last_hidden_state

        # Get encoder outputs from MagnetWhisper
        magnet_encoder_outputs = magnet_model.model.encoder(
            test_input,
            attention_mask=attention_mask,
            return_dict=True
        )
        magnet_hidden = magnet_encoder_outputs.last_hidden_state

    # Compare shapes
    print(f"\nVanilla hidden state shape: {vanilla_hidden.shape}")
    print(f"MagnetWhisper hidden state shape: {magnet_hidden.shape}")

    # Check if identical
    are_identical = torch.allclose(
        vanilla_hidden, magnet_hidden, atol=5e-3, rtol=1e-3)

    if are_identical:
        print("✓ Hidden states are IDENTICAL")
    else:
        print("✗ Hidden states are DIFFERENT")

        # Calculate difference metrics
        abs_diff = torch.abs(vanilla_hidden - magnet_hidden)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        print(f"\nDifference Statistics:")
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")

        # Show where differences are largest
        max_diff_idx = abs_diff.argmax()
        print(f"  Largest difference at index: {max_diff_idx}")

    return are_identical, vanilla_hidden


def compare_decoder_outputs(vanilla_model, magnet_model, encoder_hidden_states):
    """
    Compare decoder outputs between vanilla Whisper and MagnetWhisper.

    Args:
        vanilla_model: Vanilla Whisper model
        magnet_model: MagnetWhisper model
        encoder_hidden_states: Encoder output tensor to be used by both decoders

    Returns:
        bool: True if decoder outputs are identical, False otherwise
    """
    vanilla_model.eval()
    magnet_model.eval()

    # Create dummy decoder inputs
    decoder_input_ids = torch.tensor(
        [[1, 1]]) * vanilla_model.config.decoder_start_token_id
    decoder_input_ids = decoder_input_ids.to("cuda")

    with torch.no_grad():
        # Get decoder outputs from vanilla Whisper
        vanilla_decoder_outputs = vanilla_model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        vanilla_hidden = vanilla_decoder_outputs.last_hidden_state

        # Get decoder outputs from MagnetWhisper
        magnet_decoder_outputs = magnet_model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        magnet_hidden = magnet_decoder_outputs.last_hidden_state

    # Compare shapes
    print(f"\nVanilla decoder hidden state shape: {vanilla_hidden.shape}")
    print(
        f"MagnetWhisper decoder hidden state shape: {magnet_hidden.shape}")

    # Check if identical
    are_identical = torch.allclose(
        vanilla_hidden, magnet_hidden, atol=5e-3, rtol=1e-3)

    if are_identical:
        print("✓ Decoder hidden states are IDENTICAL")
    else:
        print("✗ Decoder hidden states are DIFFERENT")

        # Calculate difference metrics
        abs_diff = torch.abs(vanilla_hidden - magnet_hidden)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        print(f"\nDifference Statistics:")
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")

        # Show where differences are largest
        max_diff_idx = abs_diff.argmax()
        print(f"  Largest difference at index: {max_diff_idx}")

    return are_identical


def main():
    print("=" * 60)
    print("Model Comparison: Vanilla Whisper vs MagnetWhisper")
    print("=" * 60)

    # ============= Load Both Models =============
    print("\nLoading vanilla Whisper model...")
    vanilla_model = load_vanilla_whisper()
    vanilla_model.to("cuda")

    print("\nLoading MagnetWhisper (no boundary predictors)...")
    magnet_model = load_magnet_whisper(boundary_config=[])
    magnet_model.to("cuda")

    # Configure generation for both
    for model in [vanilla_model, magnet_model]:
        model.generation_config.language = "english"
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None

    print(
        f"\nVanilla model parameters: {sum(p.numel() for p in vanilla_model.parameters()) / 1e6:.1f}M")
    print(
        f"MagnetWhisper parameters: {sum(p.numel() for p in magnet_model.parameters()) / 1e6:.1f}M")

    # ============= Load Data =============
    print(f"\nLoading LibriSpeech data...")
    data_module = create_librispeech_data_module(
        vanilla_model.config.decoder_start_token_id
    )
    dataset = data_module.dataset
    processor = data_module.processor

    # ============= Test on a Single Sample =============
    print("\n" + "=" * 60)
    print("Testing Encoder Hidden States on Sample Input")
    print("=" * 60)

    # Get a test sample
    test_sample = dataset[EVAL_DATASET][0]
    test_input = torch.tensor(
        test_sample["input_features"]).unsqueeze(0).to("cuda")
    test_mask = torch.tensor(
        test_sample["attention_mask"]).unsqueeze(0).to("cuda")

    print(f"Test input shape: {test_input.shape}")

    # Compare encoder hidden states
    are_encoders_identical, vanilla_encoder_hidden_states = compare_encoder_hidden_states(
        vanilla_model,
        magnet_model,
        test_input,
        test_mask
    )

    # ============= Compare Decoder Outputs =============
    if are_encoders_identical:
        print("\n" + "=" * 60)
        print("Testing Decoder Outputs on Sample Input")
        print("=" * 60)
        compare_decoder_outputs(
            vanilla_model,
            magnet_model,
            vanilla_encoder_hidden_states
        )

    # ============= Unload Vanilla Model to Save Memory =============
    print("\n" + "=" * 60)
    print("Unloading vanilla Whisper model to free memory...")
    print("=" * 60)

    del vanilla_model
    torch.cuda.empty_cache()
    print("✓ Vanilla model unloaded")

    # ============= Full Evaluation =============
    if are_encoders_identical:
        print("\n" + "=" * 60)
        print("Encoder states match! Running full WER evaluation...")
        print("=" * 60)

        print(f"\nEvaluating MagnetWhisper...")
        magnet_metrics = evaluate_model_batched(
            magnet_model,
            dataset[EVAL_DATASET],
            processor,
            batch_size=BATCH_SIZE
        )

        # ============= Report Results =============
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"MagnetWhisper WER:       {magnet_metrics['wer']:.2f}%")
        print(f"Samples evaluated:       {magnet_metrics['num_samples']}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("WARNING: Encoder states differ!")
        print("Skipping full evaluation - fix encoder first.")
        print("=" * 60)


if __name__ == "__main__":
    main()
