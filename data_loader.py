from datasets import load_from_disk
from pathlib import Path


def prepare_dataset(batch, feature_extractor, tokenizer):
    # batch["audio"] is a list of dicts with keys 'array' and 'sampling_rate'
    audio_arrays = [audio["array"] for audio in batch["audio"]]
    sampling_rates = [audio["sampling_rate"] for audio in batch["audio"]]

    # Feature extraction on a batch
    # If sampling_rate varies in the batch, set it individually, otherwise use a fixed value
    input_features = feature_extractor(
        # assuming all have the same sampling_rate
        audio_arrays, sampling_rate=sampling_rates[0]
    ).input_features

    # Tokenize all texts in the batch
    labels = tokenizer([txt.lower() for txt in batch["text"]],
                       padding="longest", truncation=True).input_ids

    # Return a new batch dictionary
    return {
        "input_features": input_features,
        "labels": labels
    }


def get_dataset(dataset, feature_extractor, tokenizer,
                num_proc=1,
                split_name="train.clean.360",
                cache_dir="/fs/scratch/lees_stuff"):
    cache_path = Path(cache_dir) / f"{split_name}.arrow"
    if cache_path.exists():
        print("🔁 Loading processed dataset from disk...")
        return load_from_disk(cache_dir)[split_name]

    print("🔄 Processing dataset...")
    processed = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names[split_name],
        num_proc=num_proc,
        batched=True,
        fn_kwargs={
            "feature_extractor": feature_extractor,
            "tokenizer": tokenizer,
        },
    )
    processed.save_to_disk(cache_dir)
    return processed
