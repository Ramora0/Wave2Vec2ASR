from transformers import WhisperFeatureExtractor
from datasets import load_from_disk
from pathlib import Path


def prepare_dataset(batch, feature_extractor: WhisperFeatureExtractor, tokenizer):
    # batch["audio"] is a list of dicts with keys 'array' and 'sampling_rate'
    audio_arrays = [audio["array"] for audio in batch["audio"]]
    sampling_rates = [audio["sampling_rate"] for audio in batch["audio"]]

    # Feature extraction on a batch
    # If sampling_rate varies in the batch, set it individually, otherwise use a fixed value
    features = feature_extractor(
        audio_arrays, sampling_rate=sampling_rates[0], return_attention_mask=True
    )
    input_features = features.input_features
    attention_mask = features.attention_mask

    # Tokenize all texts in the batch
    normalized_texts = [txt.lower() for txt in batch["text"]]
    labels = tokenizer(normalized_texts,
                       padding="longest", truncation=True).input_ids

    # Track target boundary counts (characters excluding whitespace) so
    # downstream components can encourage matching predicted segments.
    boundary_counts = [
        sum(1 for char in text if not char.isspace()) for text in batch["text"]
    ]

    # Return a new batch dictionary
    batch_dict = {
        "input_features": input_features,
        "labels": labels,
        "attention_mask": attention_mask,
        "target_boundary_count": boundary_counts,
    }
    return batch_dict


def get_dataset(dataset, feature_extractor, tokenizer,
                num_proc=1,
                cache_dir="librispeech-full"):
    cache_path = Path(f"/fs/scratch/PAS2836/lees_stuff/{cache_dir}")
    print("🔄 Processing entire dataset...")
    processed = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc,
        batched=True,
        fn_kwargs={
            "feature_extractor": feature_extractor,
            "tokenizer": tokenizer,
        },
        load_from_cache_file=False,
    )
    print("✅ Dataset processed. Saving entire dataset to disk...")
    print(processed["train"][0].keys())
    processed.save_to_disk(cache_path)
    return processed
