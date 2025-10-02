from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Union

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from evaluate import load as load_metric
from g2p_en import G2p
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizer

from utils import recover_text_from_feature, tokenize_batch_texts


SCRATCH_PATH = "/fs/scratch/PAS2836/lees_stuff"
DEFAULT_MODEL_NAME = "openai/whisper-small"
HF_AUTH_TOKEN = "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"
DEFAULT_DATASET_NAME = "speechcolab/gigaspeech"
DEFAULT_SUBSET = "xl"
DEFAULT_RAW_DIR = "gigaspeech-xl-raw"
DEFAULT_PROCESSED_DIR = "gigaspeech-xl-processed"
TARGET_SPLITS = ("train", "validation", "test")
REQUIRED_COLUMNS = ("audio", "text")


G2P_CONVERTER = G2p()


def _normalize_text(text: str) -> str:
    return text.strip().lower()


def count_phonemes(text: str) -> float:
    normalized = _normalize_text(text)
    if not normalized:
        return 0.0
    phoneme_sequence = G2P_CONVERTER(normalized)
    phoneme_tokens = [token for token in phoneme_sequence if token.strip() and token != " "]
    return float(len(phoneme_tokens))


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor: WhisperProcessor, decoder_start_token_id: int):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        attention_masks = torch.stack([feature["attention_mask"] for feature in features])
        batch["attention_mask"] = attention_masks

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        if features:
            phoneme_counts: List[float] = []

            for idx, feature in enumerate(features):
                text = recover_text_from_feature(
                    feature,
                    labels_batch["input_ids"][idx],
                    labels_batch["attention_mask"][idx],
                    self.processor.tokenizer,
                    self.decoder_start_token_id,
                )
                phoneme_counts.append(count_phonemes(text))

            phoneme_tensor = torch.tensor(phoneme_counts, dtype=torch.float32)
            batch["target_boundary_counts"] = phoneme_tensor.unsqueeze(0)

        return batch


@dataclass
class GigaSpeechDataModule:
    dataset: DatasetDict
    tokenizer: WhisperTokenizer
    processor: WhisperProcessor
    data_collator: DataCollatorSpeechSeq2SeqWithPadding
    compute_metrics: Callable[[Any], Dict[str, float]]


def load_gigaspeech_dataset(
    scratch_path: str = SCRATCH_PATH,
    dataset_dir: str = DEFAULT_PROCESSED_DIR,
) -> DatasetDict:
    dataset_path = Path(scratch_path) / dataset_dir
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed GigaSpeech dataset not found at {dataset_path}."
            " Run preprocess_gigaspeech_dataset first."
        )

    dataset = load_from_disk(str(dataset_path))
    missing_splits = [split for split in TARGET_SPLITS if split not in dataset]
    if missing_splits:
        raise ValueError(f"Processed dataset missing required splits: {missing_splits}")

    return dataset.with_format("torch")


def load_whisper_processor(
    model_name: str = DEFAULT_MODEL_NAME, token: str = HF_AUTH_TOKEN
) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(
        model_name, language="English", task="transcribe", token=token
    )


def build_compute_metrics(tokenizer: WhisperTokenizer) -> Callable[[Any], Dict[str, float]]:
    wer_metric = load_metric("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


def create_gigaspeech_data_module(
    decoder_start_token_id: int,
    *,
    scratch_path: str = SCRATCH_PATH,
    dataset_dir: str = DEFAULT_PROCESSED_DIR,
) -> GigaSpeechDataModule:
    processor = load_whisper_processor()
    tokenizer = processor.tokenizer
    dataset = load_gigaspeech_dataset(scratch_path=scratch_path, dataset_dir=dataset_dir)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=decoder_start_token_id
    )
    compute_metrics = build_compute_metrics(tokenizer)
    return GigaSpeechDataModule(
        dataset=dataset,
        tokenizer=tokenizer,
        processor=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


def _validate_required_columns(columns: Iterable[str]) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns {missing}."
            f" Available columns: {sorted(columns)}"
        )


def _prepare_batch(
    batch: Dict[str, Any],
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
) -> Dict[str, Any]:
    audio_arrays = []

    for audio in batch["audio"]:
        if isinstance(audio, dict):
            audio_arrays.append(audio["array"])
        else:
            raise TypeError(
                "Expected audio entries to be dicts with 'array' and 'sampling_rate' keys"
            )

    target_rate = feature_extractor.sampling_rate
    features = feature_extractor(
        audio_arrays,
        sampling_rate=target_rate,
        return_attention_mask=True,
    )

    tokenized = tokenize_batch_texts(batch["text"], tokenizer)
    labels = tokenized.input_ids

    return {
        "input_features": features.input_features,
        "attention_mask": features.attention_mask,
        "labels": labels,
    }


def load_raw_gigaspeech_subset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    subset: str = DEFAULT_SUBSET,
    cache_dir: Union[str, Path, None] = None,
) -> DatasetDict:
    raw_splits: Dict[str, Dataset] = {}
    for split in TARGET_SPLITS:
        raw_splits[split] = load_dataset(
            dataset_name,
            subset,
            split=split,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )

    for split, dataset in raw_splits.items():
        _validate_required_columns(dataset.column_names)

    return DatasetDict(raw_splits)


def preprocess_gigaspeech_dataset(
    raw_dataset: DatasetDict,
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
    *,
    num_proc: int = 4,
    output_path: Union[str, Path] = Path(SCRATCH_PATH) / DEFAULT_PROCESSED_DIR,
) -> DatasetDict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for split in TARGET_SPLITS:
        if split not in raw_dataset:
            raise ValueError(f"Raw dataset missing required split '{split}'")
        _validate_required_columns(raw_dataset[split].column_names)

    print("🔄 Processing GigaSpeech dataset...")
    processed = raw_dataset.map(
        _prepare_batch,
        remove_columns=raw_dataset["train"].column_names,
        num_proc=num_proc,
        batched=True,
        fn_kwargs={
            "feature_extractor": feature_extractor,
            "tokenizer": tokenizer,
        },
        load_from_cache_file=False,
    )

    print("✅ Processing complete. Saving to disk...")
    processed.save_to_disk(str(output_path))
    return processed


def build_processed_gigaspeech(
    *,
    scratch_path: str = SCRATCH_PATH,
    raw_dir: str = DEFAULT_RAW_DIR,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    dataset_name: str = DEFAULT_DATASET_NAME,
    subset: str = DEFAULT_SUBSET,
    num_proc: int = 4,
    cache_dir: Union[str, Path, None] = None,
) -> DatasetDict:
    raw_path = Path(scratch_path) / raw_dir
    if raw_path.exists():
        raw_dataset = load_from_disk(str(raw_path))
    else:
        raw_dataset = load_raw_gigaspeech_subset(
            dataset_name=dataset_name,
            subset=subset,
            cache_dir=cache_dir,
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_dataset.save_to_disk(str(raw_path))

    processor = load_whisper_processor()

    processed_path = Path(scratch_path) / processed_dir
    processed = preprocess_gigaspeech_dataset(
        raw_dataset,
        processor.feature_extractor,
        processor.tokenizer,
        num_proc=num_proc,
        output_path=processed_path,
    )
    print(f"💾 Saved processed GigaSpeech dataset to {processed_path}")
    return processed


if __name__ == "__main__":
    build_processed_gigaspeech()
