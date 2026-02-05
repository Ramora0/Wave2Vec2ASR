"""
Standalone evaluation script for comparing MagnetWhisper vs vanilla Whisper.

This script loads a model (either MagnetWhisper or vanilla Whisper),
evaluates it on LibriSpeech validation set, and reports WER.
"""

import os
import torch
import sys
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
BATCH_SIZE = 64  # Batch size for evaluation
EVAL_DATASET = "validation.clean"  # Options: "validation.clean", "test.clean"


def manual_generate(model, input_features, attention_mask, max_new_tokens=448):
    """
    Manual generation using multi-step decoder approach (for testing).

    Args:
        model: Whisper model
        input_features: Input audio features
        attention_mask: Attention mask
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Generated token IDs
    """
    model.eval()
    batch_size = input_features.shape[0]

    # Get encoder outputs
    encoder_outputs = model.model.encoder(
        input_features,
        attention_mask=attention_mask,
        return_dict=True
    )
    encoder_hidden_states = encoder_outputs.last_hidden_state

    # Initialize with full prompt sequence
    # [50258, 50259, 50359, 50363] are the standard Whisper prompt tokens
    decoder_input_ids = torch.tensor(
        [[50258, 50259, 50359, 50363]] * batch_size,
        dtype=torch.long
    ).to(input_features.device)

    gen_config = model.generation_config
    eos_token_id = model.config.eos_token_id

    # Generate tokens one by one
    for step in range(max_new_tokens):
        # Run decoder
        decoder_outputs = model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=attention_mask,
        )
        hidden_states = decoder_outputs.last_hidden_state

        # Get logits for next token
        logits = model.get_output_embeddings()(hidden_states[:, -1, :])

        # Apply logit suppression (only on first generated token)
        if step == 0 and gen_config.begin_suppress_tokens is not None:
            for token_id in gen_config.begin_suppress_tokens:
                logits[:, token_id] = -float("inf")

        # Always suppress suppress_tokens if configured
        if gen_config.suppress_tokens is not None:
            for token_id in gen_config.suppress_tokens:
                logits[:, token_id] = -float("inf")

        # Get next token
        next_token = logits.argmax(dim=-1, keepdim=True)

        # Append to sequence
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

        # Check if all sequences have generated EOS
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return decoder_input_ids


def evaluate_model_batched(model, eval_dataset, processor, batch_size=128, use_manual_generate=False):
    """
    Evaluate model on a dataset with batched inference.

    Args:
        model: MagnetWhisper or Whisper model
        eval_dataset: Evaluation dataset
        processor: Whisper processor
        batch_size: Batch size for inference
        use_manual_generate: If True, use manual multi-step generation instead of .generate()

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
                if use_manual_generate:
                    predicted_ids = manual_generate(
                        model,
                        input_features,
                        attention_mask,
                    )
                else:
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
            token=os.environ.get("HF_TOKEN")
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
            token=os.environ.get("HF_TOKEN")
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


def compare_decoder_outputs(vanilla_model, magnet_model, encoder_hidden_states, encoder_attention_mask=None):
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
            # encoder_attention_mask=encoder_attention_mask,
        )
        vanilla_hidden = vanilla_decoder_outputs.last_hidden_state

        # Get decoder outputs from MagnetWhisper
        magnet_decoder_outputs = magnet_model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
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


def compare_decoder_outputs_multi_step(vanilla_model, magnet_model, encoder_hidden_states, encoder_attention_mask=None, num_steps=3, initial_decoder_ids=None):
    """
    Compare decoder outputs between vanilla Whisper and MagnetWhisper over multiple generation steps.
    This manually runs the decoder multiple times, appending tokens each iteration, to test if
    generation parity holds beyond a single step.

    Args:
        vanilla_model: Vanilla Whisper model
        magnet_model: MagnetWhisper model
        encoder_hidden_states: Encoder output tensor to be used by both decoders
        encoder_attention_mask: Optional attention mask for encoder states
        num_steps: Number of tokens to generate
        initial_decoder_ids: Optional initial decoder input IDs (if None, uses decoder_start_token_id)

    Returns:
        bool: True if decoder outputs are identical at each step, False otherwise
    """
    vanilla_model.eval()
    magnet_model.eval()

    # Use provided initial IDs or fall back to single decoder_start_token
    if initial_decoder_ids is not None:
        vanilla_input_ids = initial_decoder_ids.clone()
        magnet_input_ids = initial_decoder_ids.clone()
        print(
            f"Using provided initial decoder IDs: {initial_decoder_ids.tolist()}")
    else:
        vanilla_input_ids = torch.tensor(
            [[vanilla_model.config.decoder_start_token_id]], dtype=torch.long).to("cuda")
        magnet_input_ids = vanilla_input_ids.clone()
        print(
            f"Using default decoder_start_token_id: {vanilla_model.config.decoder_start_token_id}")

    # Check if model has logit suppression configured
    gen_config = vanilla_model.generation_config
    print(f"\nGeneration config info:")
    print(f"  begin_suppress_tokens: {gen_config.begin_suppress_tokens}")
    print(f"  suppress_tokens: {gen_config.suppress_tokens}")

    all_identical = True

    with torch.no_grad():
        for step in range(num_steps):
            print(f"\nStep {step + 1}/{num_steps}:")
            print(f"  Current vanilla input_ids: {vanilla_input_ids.tolist()}")
            print(f"  Current magnet input_ids:  {magnet_input_ids.tolist()}")

            # Run vanilla decoder
            vanilla_decoder_outputs = vanilla_model.model.decoder(
                input_ids=vanilla_input_ids,
                encoder_hidden_states=encoder_hidden_states,
            )
            vanilla_hidden = vanilla_decoder_outputs.last_hidden_state

            # Run magnet decoder
            magnet_decoder_outputs = magnet_model.model.decoder(
                input_ids=magnet_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            magnet_hidden = magnet_decoder_outputs.last_hidden_state

            # Compare hidden states at this step
            step_identical = torch.allclose(
                vanilla_hidden, magnet_hidden, atol=5e-3, rtol=1e-3)

            if step_identical:
                print(f"  ✓ Hidden states at step {step + 1} are IDENTICAL")
            else:
                print(f"  ✗ Hidden states at step {step + 1} are DIFFERENT")
                all_identical = False

                # Calculate difference metrics
                abs_diff = torch.abs(vanilla_hidden - magnet_hidden)
                max_diff = abs_diff.max().item()
                mean_diff = abs_diff.mean().item()

                print(f"    Max absolute difference: {max_diff:.2e}")
                print(f"    Mean absolute difference: {mean_diff:.2e}")

            # Get logits and next token from the last position
            vanilla_logits = vanilla_model.get_output_embeddings()(
                vanilla_hidden[:, -1, :])
            magnet_logits = magnet_model.get_output_embeddings()(
                magnet_hidden[:, -1, :])

            # Apply logit suppression like .generate() does
            # Suppress begin_suppress_tokens for the first generated token
            if step == 0 and gen_config.begin_suppress_tokens is not None:
                print(
                    f"  Suppressing begin tokens: {gen_config.begin_suppress_tokens[:10]}... (showing first 10)")
                for token_id in gen_config.begin_suppress_tokens:
                    vanilla_logits[:, token_id] = -float("inf")
                    magnet_logits[:, token_id] = -float("inf")

            # Always suppress suppress_tokens if configured
            if gen_config.suppress_tokens is not None:
                for token_id in gen_config.suppress_tokens:
                    vanilla_logits[:, token_id] = -float("inf")
                    magnet_logits[:, token_id] = -float("inf")

            print(f"  Vanilla logits shape: {vanilla_logits.shape}")
            print(
                f"  Vanilla logits max value: {vanilla_logits.max().item():.2f}")
            print(
                f"  Vanilla logits min value: {vanilla_logits.min().item():.2f}")

            # Get top 5 predictions to see what's happening
            vanilla_top5 = vanilla_logits.topk(5, dim=-1)
            print(f"  Vanilla top 5 tokens: {vanilla_top5.indices.tolist()}")
            print(f"  Vanilla top 5 scores: {vanilla_top5.values.tolist()}")

            vanilla_next_token = vanilla_logits.argmax(dim=-1, keepdim=True)
            magnet_next_token = magnet_logits.argmax(dim=-1, keepdim=True)

            print(f"  Vanilla next token: {vanilla_next_token.item()}")
            print(f"  Magnet next token:  {magnet_next_token.item()}")

            # Check if next tokens match
            if vanilla_next_token.item() != magnet_next_token.item():
                print(f"  ✗ Next tokens DIFFER at step {step + 1}")
                all_identical = False

            # Append the next token to input_ids for the next step
            vanilla_input_ids = torch.cat(
                [vanilla_input_ids, vanilla_next_token], dim=1)
            magnet_input_ids = torch.cat(
                [magnet_input_ids, magnet_next_token], dim=1)

    print(f"\n{'✓' if all_identical else '✗'} Multi-step decoder outputs are {'IDENTICAL' if all_identical else 'DIFFERENT'}")
    return all_identical


def main():
    print("=" * 60)
    print("Testing MagnetWhisper with explicit prompt tokens")
    print("Single-sample sanity run")
    print("=" * 60)

    # ============= Load Models =============
    # print("\nLoading vanilla Whisper model...")
    # vanilla_model = load_vanilla_whisper()
    # vanilla_model.to("cuda")

    print("\nLoading MagnetWhisper model...")
    magnet_model = load_magnet_whisper()
    magnet_model.to("cuda")

    # Configure generation for MagnetWhisper only
    magnet_model.generation_config.language = "english"
    magnet_model.generation_config.task = "transcribe"
    magnet_model.generation_config.forced_decoder_ids = None

    # Print generation configs to compare (disabled)
    # print("\n" + "=" * 60)
    # print("GENERATION CONFIG COMPARISON")
    # print("=" * 60)

    # print("\nVanilla Whisper generation_config:")
    # print(f"  decoder_start_token_id: {vanilla_model.generation_config.decoder_start_token_id}")
    # print(f"  forced_decoder_ids: {vanilla_model.generation_config.forced_decoder_ids}")
    # print(f"  language: {vanilla_model.generation_config.language}")
    # print(f"  task: {vanilla_model.generation_config.task}")
    # print(f"  begin_suppress_tokens: {vanilla_model.generation_config.begin_suppress_tokens}")
    # print(f"  suppress_tokens: {vanilla_model.generation_config.suppress_tokens}")
    # print(f"  eos_token_id: {vanilla_model.generation_config.eos_token_id}")
    # print(f"  pad_token_id: {vanilla_model.generation_config.pad_token_id}")

    # print("\nMagnet Whisper generation_config:")
    # print(f"  decoder_start_token_id: {magnet_model.generation_config.decoder_start_token_id}")
    # print(f"  forced_decoder_ids: {magnet_model.generation_config.forced_decoder_ids}")
    # print(f"  language: {magnet_model.generation_config.language}")
    # print(f"  task: {magnet_model.generation_config.task}")
    # print(f"  begin_suppress_tokens: {magnet_model.generation_config.begin_suppress_tokens}")
    # print(f"  suppress_tokens: {magnet_model.generation_config.suppress_tokens}")
    # print(f"  eos_token_id: {magnet_model.generation_config.eos_token_id}")
    # print(f"  pad_token_id: {magnet_model.generation_config.pad_token_id}")

    # print("\nVanilla model config:")
    # print(f"  decoder_start_token_id: {vanilla_model.config.decoder_start_token_id}")
    # print(f"  eos_token_id: {vanilla_model.config.eos_token_id}")
    # print(f"  pad_token_id: {vanilla_model.config.pad_token_id}")

    # print("\nMagnet model config:")
    # print(f"  decoder_start_token_id: {magnet_model.config.decoder_start_token_id}")
    # print(f"  eos_token_id: {magnet_model.config.eos_token_id}")
    # print(f"  pad_token_id: {magnet_model.config.pad_token_id}")

    # ============= Load Data =============
    print(f"\nLoading LibriSpeech data...")
    data_module = create_librispeech_data_module(
        magnet_model.config.decoder_start_token_id
    )
    dataset = data_module.dataset
    processor = data_module.processor

    # ============= Get Single Sample =============
    print("\n" + "=" * 60)
    print("Loading single test sample")
    print("=" * 60)

    # Get first sample from validation set
    sample = dataset[EVAL_DATASET][0]
    input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to("cuda")
    attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to("cuda")
    reference_text = processor.decode(sample["labels"], skip_special_tokens=True)

    # print(f"\nReference text: '{reference_text}'")

    # # ============= VANILLA WHISPER: .generate() =============
    # print("\n" + "=" * 60)
    # print("VANILLA WHISPER: .generate()")
    # print("=" * 60)

    # vanilla_model.eval()
    # with torch.no_grad():
    #     with torch.amp.autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE, device_type="cuda"):
    #         vanilla_gen_ids = vanilla_model.generate(
    #             input_features,
    #             attention_mask=attention_mask,
    #             return_dict_in_generate=True,
    #             output_scores=True,
    #         )
    #         # Extract just the sequences
    #         if hasattr(vanilla_gen_ids, 'sequences'):
    #             vanilla_gen_ids = vanilla_gen_ids.sequences

    # vanilla_gen_text = processor.decode(vanilla_gen_ids[0], skip_special_tokens=True)
    # print(f"\nVanilla .generate() output:")
    # print(f"  First 10 tokens: {vanilla_gen_ids[0][:10].tolist()}")
    # print(f"  Last 10 tokens: {vanilla_gen_ids[0][-10:].tolist()}")
    # print(f"  Total tokens: {len(vanilla_gen_ids[0])}")
    # print(f"  Text: '{vanilla_gen_text[:200]}...'")

    # # ============= VANILLA WHISPER: manual_generate() =============
    # print("\n" + "=" * 60)
    # print("VANILLA WHISPER: manual_generate()")
    # print("=" * 60)

    # with torch.no_grad():
    #     with torch.amp.autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE, device_type="cuda"):
    #         vanilla_manual_ids = manual_generate(
    #             vanilla_model,
    #             input_features,
    #             attention_mask,
    #             max_new_tokens=50,  # Limit for debugging
    #         )

    # vanilla_manual_text = processor.decode(vanilla_manual_ids[0], skip_special_tokens=True)
    # print(f"\nVanilla manual_generate() output:")
    # print(f"  All tokens: {vanilla_manual_ids[0].tolist()}")
    # print(f"  Total tokens: {len(vanilla_manual_ids[0])}")
    # print(f"  Text: '{vanilla_manual_text}'")

    # # ============= MAGNET WHISPER: .generate() WITHOUT prompt tokens =============
    # print("\n" + "=" * 60)
    # print("MAGNET WHISPER: .generate() WITHOUT explicit prompt tokens")
    # print("=" * 60)

    # magnet_model.eval()
    # with torch.no_grad():
    #     with torch.amp.autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE, device_type="cuda"):
    #         magnet_gen_ids_no_prompt = magnet_model.generate(
    #             input_features,
    #             attention_mask=attention_mask,
    #             max_new_tokens=50,  # Limit to prevent infinite loop
    #         )

    # magnet_gen_text_no_prompt = processor.decode(magnet_gen_ids_no_prompt[0], skip_special_tokens=True)
    # print(f"\nMagnet .generate() WITHOUT prompt output:")
    # print(f"  All tokens: {magnet_gen_ids_no_prompt[0].tolist()}")
    # print(f"  Total tokens: {len(magnet_gen_ids_no_prompt[0])}")
    # print(f"  Text: '{magnet_gen_text_no_prompt}'")

    # ============= MAGNET WHISPER: .generate() WITH manual prompt tokens =============
    print("\n" + "=" * 60)
    print("MAGNET WHISPER: .generate() WITH explicit prompt tokens")
    print("=" * 60)

    # Manually provide the prompt tokens that Whisper should use
    # [50258, 50259, 50359, 50363] = [<|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>]
    prompt_tokens = torch.tensor(
        [[50258, 50259, 50359, 50363]],
        dtype=torch.long
    ).to("cuda")

    print(f"\nManually providing prompt tokens: {prompt_tokens[0].tolist()}")

    magnet_model.eval()
    with torch.no_grad():
        with torch.amp.autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE, device_type="cuda"):
            magnet_gen_ids_with_prompt = magnet_model.generate(
                input_features,
                attention_mask=attention_mask,
                decoder_input_ids=prompt_tokens,
                max_new_tokens=50,
            )

    magnet_gen_text_with_prompt = processor.decode(magnet_gen_ids_with_prompt[0], skip_special_tokens=True)
    print(f"\nMagnet .generate() WITH prompt output:")
    print(f"  All tokens: {magnet_gen_ids_with_prompt[0].tolist()}")
    print(f"  Total tokens: {len(magnet_gen_ids_with_prompt[0])}")
    print(f"  Text: '{magnet_gen_text_with_prompt}'")

    # ============= FULL DATASET WER EVALUATION =============
    print("\n" + "=" * 60)
    print("FULL DATASET WER EVALUATION - MagnetWhisper")
    print("=" * 60)
    print(f"\nEvaluating on {EVAL_DATASET} with batch size {BATCH_SIZE}...")

    eval_dataset = dataset[EVAL_DATASET]

    # Evaluate using .generate()
    print("\nRunning full evaluation with .generate()...")
    metrics = evaluate_model_batched(
        magnet_model,
        eval_dataset,
        processor,
        batch_size=BATCH_SIZE,
        use_manual_generate=False
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Dataset: {EVAL_DATASET}")
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"Word Error Rate (WER): {metrics['wer']:.2f}%")
    print("=" * 60)

    # # ============= MAGNET WHISPER: manual_generate() =============
    # print("\n" + "=" * 60)
    # print("MAGNET WHISPER: manual_generate()")
    # print("=" * 60)

    # with torch.no_grad():
    #     with torch.amp.autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE, device_type="cuda"):
    #         magnet_manual_ids = manual_generate(
    #             magnet_model,
    #             input_features,
    #             attention_mask,
    #             max_new_tokens=50,  # Limit for debugging
    #         )

    # magnet_manual_text = processor.decode(magnet_manual_ids[0], skip_special_tokens=True)
    # print(f"\nMagnet manual_generate() output:")
    # print(f"  All tokens: {magnet_manual_ids[0].tolist()}")
    # print(f"  Total tokens: {len(magnet_manual_ids[0])}")
    # print(f"  Text: '{magnet_manual_text}'")

    # # ============= COMPARISON SUMMARY =============
    # print("\n" + "=" * 60)
    # print("COMPARISON SUMMARY")
    # print("=" * 60)

    # print(f"\nReference: '{reference_text}'")
    # print(f"\nVanilla Whisper:")
    # print(f"  .generate():       '{vanilla_gen_text[:100]}...'")
    # print(f"  manual_generate(): '{vanilla_manual_text}'")

    # print(f"\nMagnet Whisper:")
    # print(f"  .generate() WITHOUT prompt: '{magnet_gen_text_no_prompt}'")
    # print(f"  .generate() WITH prompt:    '{magnet_gen_text_with_prompt}'")
    # print(f"  manual_generate():          '{magnet_manual_text}'")

    # # Check if vanilla methods match
    # vanilla_match = vanilla_gen_text.strip() == vanilla_manual_text.strip()
    # print(f"\nVanilla methods match: {'✓ YES' if vanilla_match else '✗ NO'}")

    # # Check if magnet with prompt matches manual
    # magnet_match_with_prompt = magnet_gen_text_with_prompt.strip() == magnet_manual_text.strip()
    # print(f"Magnet .generate(WITH prompt) matches manual_generate(): {'✓ YES' if magnet_match_with_prompt else '✗ NO'}")

    # # Check if magnet without prompt is broken
    # magnet_match_without_prompt = magnet_gen_text_no_prompt.strip() == magnet_manual_text.strip()
    # print(f"Magnet .generate(WITHOUT prompt) matches manual_generate(): {'✓ YES (unexpected!)' if magnet_match_without_prompt else '✗ NO (expected - this is the bug!)'}")

    # print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
