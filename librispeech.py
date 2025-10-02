from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import torch
from datasets import DatasetDict, load_from_disk
from evaluate import load as load_metric
from g2p_en import G2p
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizer

from utils import recover_text_from_feature, tokenize_batch_texts


SCRATCH_PATH = "/fs/scratch/PAS2836/lees_stuff"
DEFAULT_MODEL_NAME = "openai/whisper-small"
HF_AUTH_TOKEN = "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"


G2P_CONVERTER = G2p()


def count_phonemes(text: str) -> float:
    normalized = text.strip().lower()
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
class LibriSpeechDataModule:
    dataset: DatasetDict
    tokenizer: WhisperTokenizer
    processor: WhisperProcessor
    data_collator: DataCollatorSpeechSeq2SeqWithPadding
    compute_metrics: Callable[[Any], Dict[str, float]]


def load_librispeech_dataset(scratch_path: str = SCRATCH_PATH, dataset_dir: str = "librispeech-processed") -> DatasetDict:
    dataset = load_from_disk(f"{scratch_path}/{dataset_dir}")
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


def create_librispeech_data_module(decoder_start_token_id: int) -> LibriSpeechDataModule:
    processor = load_whisper_processor()
    tokenizer = processor.tokenizer
    dataset = load_librispeech_dataset()
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=decoder_start_token_id)
    compute_metrics = build_compute_metrics(tokenizer)
    return LibriSpeechDataModule(
        dataset=dataset,
        tokenizer=tokenizer,
        processor=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


def build_map_to_pred(model, processor: WhisperProcessor):
    def map_to_pred(batch):
        input_features = torch.tensor(batch["input_features"]).unsqueeze(0).to("cuda")

        batch["reference"] = processor.tokenizer._normalize(processor.decode(batch["labels"]))

        with torch.no_grad():
            predicted_ids = model.generate(input_features)[0]
        transcription = processor.decode(predicted_ids)
        batch["prediction"] = processor.tokenizer._normalize(transcription)
        return batch

    return map_to_pred


def test_inference_speed(model, dataset_subset, num_samples: int = 100):
    import time

    model.eval()
    times = []

    samples = []
    for i, sample in enumerate(dataset_subset):
        if i >= num_samples:
            break
        input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to("cuda")
        samples.append(input_features)

    if not samples:
        return 0.0, 0.0

    print(f"Testing inference speed on {len(samples)} samples...")

    with torch.no_grad():
        for _ in range(5):
            _ = model.generate(samples[0])

    with torch.no_grad():
        for input_features in samples:
            start_time = time.time()
            _ = model.generate(input_features)
            end_time = time.time()
            times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    total_time = sum(times)

    print(f"Average inference time per sample: {avg_time:.4f} seconds")
    print(f"Total inference time for {len(samples)} samples: {total_time:.4f} seconds")
    print(f"Samples per second: {len(samples) / total_time:.2f}")

    return avg_time, total_time


def _prepare_batch(
    batch: Dict[str, Any],
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
) -> Dict[str, Any]:
    audio_arrays = [audio["array"] for audio in batch["audio"]]
    sampling_rates = [audio["sampling_rate"] for audio in batch["audio"]]

    features = feature_extractor(
        audio_arrays,
        sampling_rate=sampling_rates[0],
        return_attention_mask=True,
    )

    tokenized = tokenize_batch_texts(batch["text"], tokenizer)
    labels = tokenized.input_ids

    return {
        "input_features": features.input_features,
        "attention_mask": features.attention_mask,
        "labels": labels,
    }


def preprocess_librispeech_dataset(
    raw_dataset: DatasetDict,
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
    *,
    num_proc: int = 4,
    output_path: Union[str, Path] = Path(SCRATCH_PATH) / "librispeech-processed",
) -> DatasetDict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("🔄 Processing LibriSpeech dataset...")
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


def build_processed_librispeech(
    *,
    scratch_path: str = SCRATCH_PATH,
    raw_dir: str = "librispeech-full",
    processed_dir: str = "librispeech-processed",
    num_proc: int = 4,
) -> DatasetDict:
    raw_path = Path(scratch_path) / raw_dir
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw LibriSpeech dataset not found at {raw_path}."
            " Download and cache it before running this script."
        )

    raw_dataset = load_from_disk(str(raw_path))
    processor = load_whisper_processor()

    processed_path = Path(scratch_path) / processed_dir
    processed = preprocess_librispeech_dataset(
        raw_dataset,
        processor.feature_extractor,
        processor.tokenizer,
        num_proc=num_proc,
        output_path=processed_path,
    )
    print(f"💾 Saved processed LibriSpeech dataset to {processed_path}")
    return processed


if __name__ == "__main__":
    build_processed_librispeech()
