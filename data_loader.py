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


def get_dataset(dataset, feature_extractor, tokenizer, split_name="train.clean.360"):
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names[split_name],
        num_proc=1,
        batched=True,
        fn_kwargs={
            "feature_extractor": feature_extractor,
            "tokenizer": tokenizer,
        },
        # keep_in_memory=True
    )
    return dataset
