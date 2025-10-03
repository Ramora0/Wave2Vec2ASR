from transformers import TrainerCallback
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
# from SlidingWhisper import SlidingWhisper
from evaluate import load
from MagnetWhisper import MagnetWhisper
from transformers import WhisperForConditionalGeneration

import wandb

from librispeech import (
    build_map_to_pred,
    create_librispeech_data_module,
    test_inference_speed,
)

print("hi")

# Load the model

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small",
    # torch_dtype=torch.float16,
    token="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"
    # attn_implementation="flash_attention_2"
)

# Convert to the custom MagnetWhisper stack and enable BoundaryPredictor2.
model.__class__ = MagnetWhisper
SYLLABLE_BOUNDARY_LAYER = 1
SYLLABLE_BOUNDARY_PRIOR = 0.25
BOUNDARY_TEMP_START = 2.5
BOUNDARY_TEMP_END = 1.1
BOUNDARY_TARGET_MIX_START = 0.0
BOUNDARY_TARGET_MIX_END = 1.0
boundary_priors = [(SYLLABLE_BOUNDARY_LAYER, SYLLABLE_BOUNDARY_PRIOR)]
model.load_magnet(boundary_priors, "BoundaryPredictor2")


def _set_boundary_temperature(magnet_model, temperature):
    predictors = getattr(magnet_model.model.encoder, "boundary_predictors", [])
    for predictor in predictors:
        if hasattr(predictor, "temp"):
            predictor.temp = temperature
    magnet_model.boundary_temperature = temperature


def _set_boundary_target_mix(magnet_model, mix):
    if hasattr(magnet_model, "set_boundary_target_mix"):
        magnet_model.set_boundary_target_mix(mix)
    else:
        magnet_model.boundary_target_mix = mix


_set_boundary_temperature(model, BOUNDARY_TEMP_START)
_set_boundary_target_mix(model, BOUNDARY_TARGET_MIX_START)

model.to("cuda")

model.generation_config.language = "english"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None

data_module = create_librispeech_data_module(
    model.config.decoder_start_token_id)
dataset = data_module.dataset
tokenizer = data_module.tokenizer
processor = data_module.processor
data_collator = data_module.data_collator
compute_metrics = data_module.compute_metrics

os.environ["WANDB_PROJECT"] = "whisper-magnet-osc"

MODEL_NAME = "magnet-phonemes"

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
    num_train_epochs=3,
    eval_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=16000,
    save_total_limit=2,
    eval_steps=4000,
    logging_steps=50,
    report_to="wandb",
    greater_is_better=False,
    weight_decay=.005,

    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    max_grad_norm=2.0,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation.clean"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


class CompressionRatioCallback(TrainerCallback):
    """Callback to log compression ratios to wandb during training"""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        compression_ratio = model.get_and_reset_compression_ratio()
        boundary_loss = model.get_and_reset_boundary_loss()

        log_payload = {
            "train/compression_ratio": compression_ratio
        }

        if boundary_loss is not None:
            log_payload["train/boundary_loss"] = boundary_loss

        boundary_temp = getattr(model, "boundary_temperature", None)
        if boundary_temp is not None:
            log_payload["train/boundary_temperature"] = boundary_temp

        boundary_target_mix = getattr(model, "boundary_target_mix", None)
        if boundary_target_mix is not None:
            log_payload["train/boundary_target_mix"] = boundary_target_mix

        wandb.log(log_payload)


# Add compression ratio callback
trainer.add_callback(CompressionRatioCallback())


class BoundaryScheduler(TrainerCallback):
    def __init__(self, start_temp, end_temp, start_mix, end_mix):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.start_mix = start_mix
        self.end_mix = end_mix

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        total_steps = state.max_steps if state and state.max_steps else None
        if not total_steps or total_steps <= 0:
            return

        progress = min(1.0, state.global_step / total_steps)
        current_temp = self.start_temp + \
            (self.end_temp - self.start_temp) * progress
        current_mix = self.start_mix + \
            (self.end_mix - self.start_mix) * progress

        _set_boundary_temperature(model, current_temp)
        _set_boundary_target_mix(model, current_mix)


trainer.add_callback(
    BoundaryScheduler(
        start_temp=BOUNDARY_TEMP_START,
        end_temp=BOUNDARY_TEMP_END,
        start_mix=BOUNDARY_TARGET_MIX_START,
        end_mix=BOUNDARY_TARGET_MIX_END,
    )
)

# class EvaluateFirstStepCallback(TrainerCallback):
#     def on_step_begin(self, args, state, control, **kwargs):
#         if state.global_step == 1:
#             control.should_evaluate = True

# trainer.add_callback(EvaluateFirstStepCallback())

trainer.train()
# trainer.save_model(f"./models/{MODEL_NAME}")

model.save_pretrained(f"./models/magnet-phonemes")
# model = MagnetWhisper.from_pretrained(f"./models/{MODEL_NAME}")
model = model.to("cuda")

map_to_pred = build_map_to_pred(model, processor)
avg_time, total_time = test_inference_speed(
    model,
    dataset["test.clean"],
    num_samples=1000,
)

# Then do WER evaluation separately if needed
print("\nCalculating WER...")
result = dataset["test.clean"].map(map_to_pred)

client = load("wer")
wer = client.compute(
    references=result['reference'], predictions=result['prediction'])
print(
    f"WER: {100 * wer:.2f}%")

wandb.log({
    "evaluation/wer": 100 * wer
})
