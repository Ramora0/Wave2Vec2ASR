import time
import torch
from TestMagnetWhisper import TestMagnetWhisper
from transformers import WhisperForConditionalGeneration
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from datasets import load_from_disk
import os

# Set up paths
scratch_path = "/fs/scratch/PAS2836/lees_stuff"
test_model_path = "./models/hnet-4x"

print("Loading dataset...")
dataset = load_from_disk(f"{scratch_path}/librispeech-processed")

print("Loading processor...")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe",
                                             token="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")

print("Loading MagnetWhisper model from saved path...")
# Load the model directly from the saved path
loaded_model = TestMagnetWhisper.from_pretrained(test_model_path)
loaded_model = loaded_model.to("cuda")
loaded_model.config._attn_implementation = "eager"

print("Running inference on 5 train samples...")


def run_inference_test(model, dataset_subset, num_samples=5):
    """Run inference on a subset of samples and measure performance"""
    model.eval()
    times = []

    # Get samples from train split
    samples = []
    for i, sample in enumerate(dataset_subset):
        if i >= num_samples:
            break
        input_features = torch.tensor(
            sample["input_features"]).unsqueeze(0).to("cuda")
        attention_mask = torch.tensor(
            sample["attention_mask"]).unsqueeze(0).to("cuda")
        samples.append((input_features, attention_mask))

    print(f"Testing inference on {len(samples)} samples from train split...")

    # Run inference and measure time
    predictions = []
    references = []
    boundary_data = []

    with torch.no_grad():
        for i, (input_features, attention_mask) in enumerate(samples):
            print(f"\nProcessing sample {i+1}...")

            # Clear any previous boundary positions
            model.clear_boundary_positions()

            start_time = time.time()
            predicted_ids = model.generate(
                input_features, attention_mask=attention_mask)[0]
            end_time = time.time()

            # Get boundary positions from the model
            sample_boundaries = model.get_boundary_positions()

            times.append(end_time - start_time)

            # Decode prediction and reference
            transcription = processor.decode(
                predicted_ids, skip_special_tokens=True)
            predictions.append(transcription)

            # Get reference from original dataset
            reference = processor.tokenizer._normalize(
                processor.decode(dataset_subset[i]['labels']))
            references.append(reference)

            # Store boundary data for this sample with additional info
            boundary_data_with_stats = {}
            for layer_idx, positions in sample_boundaries.items():
                # Get compression ratios from the encoder
                compression_ratio = model.model.encoder.compression_ratios.get(
                    layer_idx, 0.0)
                boundary_data_with_stats[layer_idx] = {
                    'positions': positions,
                    'num_boundaries': sum(len(pos_list) for pos_list in positions),
                    'compression_ratio': compression_ratio
                }

            boundary_data.append(boundary_data_with_stats)

            # Print boundary information for this sample
            # print(f"Sample {i+1} boundary analysis:")
            # for layer_idx in sorted(sample_boundaries.keys()):
            #     boundary_info = boundary_data_with_stats[layer_idx]
            #     print(
            #         f"  Layer {layer_idx}: {boundary_info['num_boundaries']} boundaries")
            #     print(
            #         f"    Compression ratio: {boundary_info['compression_ratio']:.4f}")
            #     print(f"    Boundary positions: {boundary_info['positions']}")

    avg_time = sum(times) / len(times)
    total_time = sum(times)

    print(f"\nInference Results:")
    print(f"Average inference time per sample: {avg_time:.4f} seconds")
    print(
        f"Total inference time for {len(samples)} samples: {total_time:.4f} seconds")
    print(f"Samples per second: {len(samples) / total_time:.2f}")

    # Show a few examples
    print(f"\nSample predictions:")
    for i in range(min(5, len(predictions))):
        print(f"Sample {i+1}:")
        print(f"  Input shape:  {samples[i][0].shape}")
        print(f"  Reference:  {references[i]}")
        print(f"  Prediction: {predictions[i]}")
        print()

    # Get compression ratio if available
    if hasattr(model, 'get_and_reset_compression_ratio'):
        compression_ratio = model.get_and_reset_compression_ratio()
        print(f"Overall compression ratio: {compression_ratio:.4f}")

    return avg_time, total_time, predictions, references, boundary_data


# Run the test
avg_time, total_time, predictions, references, boundary_data = run_inference_test(
    loaded_model, dataset["validation.clean"], num_samples=5)

print("Test completed successfully!")
print(f"Model successfully loaded from: {test_model_path}")

# Summary of boundary analysis across all samples
print("\n" + "="*60)
print("BOUNDARY ANALYSIS SUMMARY")
print("="*60)

# Analyze boundary patterns across samples
layer_boundary_counts = {}
layer_compression_ratios = {}

for sample_idx, sample_boundaries in enumerate(boundary_data):
    print(f"\nSample {sample_idx + 1} detailed boundary positions:")
    for layer_idx in sorted(sample_boundaries.keys()):
        info = sample_boundaries[layer_idx]
        if layer_idx not in layer_boundary_counts:
            layer_boundary_counts[layer_idx] = []
            layer_compression_ratios[layer_idx] = []

        layer_boundary_counts[layer_idx].append(info['num_boundaries'])
        layer_compression_ratios[layer_idx].append(info['compression_ratio'])

        print(f"  Layer {layer_idx}:")
        print(
            f"    Boundaries: {info['num_boundaries']} (ratio: {info['compression_ratio']:.4f})")
        print(f"    Positions: {info['positions']}")

# Print average statistics per layer
print(f"\nAverage boundary statistics per layer:")
for layer_idx in sorted(layer_boundary_counts.keys()):
    avg_boundaries = sum(
        layer_boundary_counts[layer_idx]) / len(layer_boundary_counts[layer_idx])
    avg_compression = sum(
        layer_compression_ratios[layer_idx]) / len(layer_compression_ratios[layer_idx])
    print(
        f"  Layer {layer_idx}: Avg {avg_boundaries:.1f} boundaries, Avg compression ratio: {avg_compression:.4f}")

print("\n" + "="*60)
