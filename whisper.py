import time
from transformers import TrainerCallback
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
# from SlidingWhisper import SlidingWhisper
from evaluate import load
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import torch
from MagnetWhisper import MagnetWhisper
from transformers import WhisperForConditionalGeneration
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from datasets import load_dataset, Audio, load_from_disk

from data_loader import get_dataset
import wandb
import aiohttp

scratch_path = "/fs/scratch/PAS2836/lees_stuff"

print("hi")
# dataset = load_dataset("openslr/librispeech_asr", trust_remote_code=True, storage_options={
#                        'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=60*60*24)}})

# dataset = dataset.remove_columns(["file", "speaker_id", "chapter_id", "id"])
# dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# # del dataset["train.clean.360"]
# del dataset["train.clean.100"]
# del dataset["train.other.500"]
# del dataset["test.other"]
# del dataset["validation.other"]

# dataset.save_to_disk(f"{scratch_path}/librispeech-trimmed")

dataset = load_from_disk(f"{scratch_path}/librispeech-full")

print(dataset["train.clean.360"][0])


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small",
                                                            token="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")

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe",
                                             token="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")

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe",
                                             token="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")

dataset = get_dataset(dataset, feature_extractor,
                      tokenizer, split_name="train.clean.360")


dataset = dataset.with_format("torch")

# Load the model

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small",
    # torch_dtype=torch.float16,
    token="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"
    # attn_implementation="flash_attention_2"
)

model.to("cuda")

# use_flash_attention_2=True)
# attn_implementation="flash_attention_2")

# model = WhisperForConditionalGeneration.from_pretrained("data/models/whisper/base")


model.generation_config.language = "english"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None

model.__class__ = MagnetWhisper
model.load_magnet([(1, 0.25)], "BoundaryPredictor2")

# model.__class__ = SlidingWhisper
# model.load_sliding(window_size=128)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


wer_metric = load("wer")


class CompressionRatioCallback(TrainerCallback):
    """Callback to log compression ratios to wandb during training"""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        compression_ratio = model.get_and_reset_compression_ratio()
        wandb.log({
            "train/compression_ratio": compression_ratio
        })


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


os.environ["WANDB_PROJECT"] = "whisper-magnet-osc"

MODEL_NAME = "magnet"

training_args = Seq2SeqTrainingArguments(
    # change to a repo name of your choice
    output_dir=f"./models/{MODEL_NAME}",

    per_device_train_batch_size=16,
    per_device_eval_batch_size=128,

    fp16=True,
    fp16_full_eval=True,
    # bf16=True,
    # bf16_full_eval=True,

    learning_rate=1e-5,
    warmup_ratio=0.1,
    # max_steps=16000,
    num_train_epochs=6,
    eval_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=16000,
    save_total_limit=2,
    eval_steps=2000,
    logging_steps=50,
    report_to="wandb",
    greater_is_better=False,
    weight_decay=.005,

    dataloader_num_workers=8,
    dataloader_pin_memory=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train.clean.360"],
    eval_dataset=dataset["validation.clean"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Add compression ratio callback
trainer.add_callback(CompressionRatioCallback())


# class EvaluateFirstStepCallback(TrainerCallback):
#     def on_step_begin(self, args, state, control, **kwargs):
#         if state.global_step == 1:
#             control.should_evaluate = True


# trainer.add_callback(EvaluateFirstStepCallback())

trainer.train()
# trainer.save_model(f"./models/{MODEL_NAME}")

# model.save_pretrained(f"./models/{MODEL_NAME}")
# model = MagnetWhisper.from_pretrained(f"./models/{MODEL_NAME}")
model = model.to("cuda")


def map_to_pred(batch):
    input_features = torch.tensor(batch["input_features"]).unsqueeze(
        0).to("cuda")  # Add batch dimension if missing

    batch["reference"] = processor.tokenizer._normalize(
        processor.decode(batch['labels']))

    with torch.no_grad():
        predicted_ids = model.generate(input_features)[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    return batch


def test_inference_speed(dataset_subset, num_samples=100):
    """Test pure inference speed without WER calculation overhead"""
    model.eval()
    times = []

    # Pre-load samples to avoid dataloading overhead
    samples = []
    for i, sample in enumerate(dataset_subset):
        if i >= num_samples:
            break
        input_features = torch.tensor(
            sample["input_features"]).unsqueeze(0).to("cuda")
        samples.append(input_features)

    print(f"Testing inference speed on {len(samples)} samples...")

    # Warm up GPU
    with torch.no_grad():
        for _ in range(5):
            _ = model.generate(samples[0])

    # Measure inference time
    with torch.no_grad():
        for input_features in samples:
            start_time = time.time()
            _ = model.generate(input_features)
            end_time = time.time()
            times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    total_time = sum(times)

    print(f"Average inference time per sample: {avg_time:.4f} seconds")
    print(
        f"Total inference time for {len(samples)} samples: {total_time:.4f} seconds")
    print(f"Samples per second: {len(samples) / total_time:.2f}")

    # wandb.log({
    #     "evaluation/avg_inference_time": avg_time,
    #     "evaluation/total_inference_time": total_time,
    #     "evaluation/samples_per_second": len(samples) / total_time
    # })

    return avg_time, total_time


# Test inference speed first
avg_time, total_time = test_inference_speed(
    dataset["validation.clean"], num_samples=1000)

# Then do WER evaluation separately if needed
print("\nCalculating WER...")
result = dataset["validation.clean"].map(map_to_pred)

client = load("wer")
wer = client.compute(
    references=result['reference'], predictions=result['prediction'])
print(
    f"WER: {100 * wer:.2f}%")

# wandb.log({
#     "evaluation/wer": 100 * wer
# })
