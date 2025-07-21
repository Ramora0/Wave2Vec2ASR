import functools
import os

def prepare_dataset(batch, feature_extractor, tokenizer):
    audio = batch["audio"]

    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

def prep_data(dataset, feature_extractor, tokenizer):
    prepare_dataset_with_args = functools.partial(prepare_dataset, feature_extractor=feature_extractor, tokenizer=tokenizer)
    dataset = dataset.map(prepare_dataset_with_args, remove_columns=dataset.column_names["train.clean.360"], num_proc=1)
    
    dataset = dataset.with_format("torch")
    return dataset