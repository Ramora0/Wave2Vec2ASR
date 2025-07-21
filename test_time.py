import time
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from MagnetWav2Vec2 import MagnetWav2Vec2

# --- Configuration ---
MODEL_PATH = "./data/models/magnet-0.67-v2-3k"
DATASET_PATH = "./data/librispeech_processed"
VOCAB_PATH = "./data/vocab.json"
BATCH_SIZE = 4 # Adjust based on GPU memory
NUM_BATCHES_TO_TIME = 250 # Number of batches to include in timing measurement
NUM_WARMUP_BATCHES = 2
USE_GPU_EVENTS = torch.cuda.is_available() # Use CUDA events if GPU is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- End Configuration ---

# 1. Load Model
print(f"Loading model from {MODEL_PATH}...")
# model = Wav2Vec2ForCTC.from_pretrained(
#     MODEL_PATH,
#     ctc_loss_reduction="mean",
#     pad_token_id=28 # Assuming pad token ID is 28 based on original code
# ).to(DEVICE).eval() # Move to device and set to evaluation mode

model = MagnetWav2Vec2.from_pretrained(
    MODEL_PATH
).to(DEVICE).eval() # Move to device and set to evaluation mode

# 2. Load Processor (Tokenizer and Feature Extractor)
print("Loading processor...")
tokenizer = Wav2Vec2CTCTokenizer(VOCAB_PATH, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True) # Ensure attention mask is returned
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# 3. Load Dataset
print(f"Loading dataset from {DATASET_PATH}...")
dataset = load_from_disk(DATASET_PATH)["test.clean"]

# 4. Define Collate Function (Handles batching and padding)
def collate_fn(batch):
    input_features = [{"input_values": sample["input_values"]} for sample in batch]
    # Pad input audio sequences and create attention mask
    batch_proc = processor.pad(
        input_features,
        padding=True,
        return_tensors="pt",
    )
    return batch_proc

# 5. Create DataLoader
print("Creating DataLoader...")
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# 6. Pre-load batches (Optional, to isolate model inference time)
print(f"Pre-loading {NUM_BATCHES_TO_TIME} batches...")
preloaded_batches = []
for i, batch in enumerate(tqdm(data_loader, total=NUM_BATCHES_TO_TIME, desc="Pre-loading Batches")):
    if i >= NUM_BATCHES_TO_TIME:
        break
    # Move batch to device during pre-loading if desired, or keep it here
    preloaded_batches.append({k: v.to(DEVICE) for k, v in batch.items()})

# 7. Timing Inference (using pre-loaded batches)
print(f"Starting timing for {len(preloaded_batches)} pre-loaded batches...")
total_time = 0.0
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    # Warmup (Optional but recommended)
    if len(preloaded_batches) > NUM_WARMUP_BATCHES:
        print(f"Running {NUM_WARMUP_BATCHES} warmup batches...")
        for i in range(NUM_WARMUP_BATCHES):
             batch = preloaded_batches[i]
             input_values = batch["input_values"]
             # attention_mask = batch.get("attention_mask") # Already on DEVICE if moved during pre-load
             _ = model(input_values)#, attention_mask=attention_mask)
        # Ensure warmup is finished before starting timer
        torch.cuda.synchronize(DEVICE)
        print("Warmup complete.")
        batches_to_time = preloaded_batches[NUM_WARMUP_BATCHES:]
    else:
        print("Not enough batches for warmup, timing all pre-loaded batches.")
        batches_to_time = preloaded_batches

    # Actual timing loop
    for batch in tqdm(batches_to_time, desc="Timing Inference"):
        input_values = batch["input_values"]
        # attention_mask = batch.get("attention_mask") # Already on DEVICE if moved during pre-load

        starter.record()
        _ = model(input_values)#, attention_mask=attention_mask)
        ender.record()
        torch.cuda.synchronize(DEVICE) # Wait for the events to complete
        batch_time = starter.elapsed_time(ender) / 1000.0 # Time in seconds
        total_time += batch_time

actual_batches_timed = len(batches_to_time)
if actual_batches_timed > 0:
    avg_time_per_batch = total_time / actual_batches_timed
    samples_per_second = BATCH_SIZE / avg_time_per_batch
else:
    avg_time_per_batch = 0
    samples_per_second = 0

print(f"\n--- Inference Timing Results ---")
print(f"Device: {DEVICE}")
print(f"Timed {actual_batches_timed} batches (after {NUM_WARMUP_BATCHES} warmup batches).")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Average time per batch: {avg_time_per_batch:.4f} seconds")
print(f"Average throughput: {samples_per_second:.2f} samples/second")

# ... rest of your script if any ...