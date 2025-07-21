import os
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
from MagnetWav2Vec2 import MagnetWav2Vec2
from datasets import load_from_disk
from evaluate import load
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm # Optional: Add import for progress bar

def get_wer(model):
    # Load components
    dataset = load_from_disk("./data/librispeech_processed")["test.clean"]
    wer_metric = load("wer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval() # Set model to evaluation mode

    from transformers import Wav2Vec2CTCTokenizer

    tokenizer = Wav2Vec2CTCTokenizer("./data/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Define a collate function to handle batching and padding
    def collate_fn(batch):
        # Assuming dataset items have 'input_values' (as list/array) and 'labels'
        # Wav2Vec2 expects a list of single audio arrays as input
        input_features = [{"input_values": sample["input_values"]} for sample in batch]
        label_features = [{"input_ids": sample["labels"]} for sample in batch]

        # Pad input audio sequences
        batch_proc = processor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        # Pad label sequences
        with processor.as_target_processor():
            labels_batch = processor.pad(
                label_features,
                padding=True,
                return_tensors="pt",
            )

        # Replace padding token id in labels with -100 for loss calculation (though not strictly needed for inference)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch_proc["labels"] = labels
        return batch_proc

    # Create DataLoader (adjust batch_size based on your GPU memory)
    # Ensure your dataset samples have 'input_values' and 'labels' keys
    data_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    all_predictions = []
    all_references = []

    # Perform inference
    print("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(data_loader): # Wrap data_loader with tqdm for progress
            input_values = batch["input_values"].to(device)
            # Handle potential absence of attention_mask if padding strategy doesn't create it
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            labels = batch["labels"] # Keep labels on CPU for decoding

            # Get model predictions (logits)
            logits = model(input_values, attention_mask=attention_mask).logits

            # Get the most likely token IDs
            predicted_ids = torch.argmax(logits, dim=-1)

            # Decode predicted IDs to text
            # Move predicted_ids to CPU for decoding if it's not already
            predictions = processor.batch_decode(predicted_ids.cpu().numpy())

            # Decode label IDs to text, handling -100 padding
            labels[labels == -100] = processor.tokenizer.pad_token_id
            references = processor.batch_decode(labels.cpu().numpy(), group_tokens=False)
            
            # print(f"Predictions: {predictions}")
            # print(f"References: {references}")

            # os._exit(0)

            all_predictions.extend(predictions)
            all_references.extend(references)

    # Compute WER
    print("Calculating WER...")
    wer = wer_metric.compute(predictions=all_predictions, references=all_references)

    print(f"Test WER: {100*wer:.1f}")