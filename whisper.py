from transformers import TrainerCallback
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.trainer import Trainer
import os
import json
from pathlib import Path
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
SYLLABLE_BOUNDARY_LAYER = 3
SYLLABLE_COMPRESSION_TARGET = 12.0
SYLLABLE_BOUNDARY_PRIOR = 1.0 / SYLLABLE_COMPRESSION_TARGET
BOUNDARY_TEMP = 1.1  # Final temperature we keep fixed during this run
# Max compression, i.e., syllable target throughout training
BOUNDARY_TARGET_PROGRESS = 1.0
FREEZE_NON_BOUNDARY_STEPS = 250
# DOWNSAMPLE_NO_GRAD_STEPS = 17600
boundary_priors = [(SYLLABLE_BOUNDARY_LAYER, SYLLABLE_BOUNDARY_PRIOR)]
model.load_magnet(boundary_priors, "BoundaryPredictor1")


def _set_boundary_temperature(magnet_model, temperature):
    predictors = getattr(magnet_model.model.encoder, "boundary_predictors", [])
    for predictor in predictors:
        if hasattr(predictor, "temp"):
            predictor.temp = temperature
    magnet_model.boundary_temperature = temperature


def _set_boundary_target_progress(magnet_model, progress):
    if hasattr(magnet_model, "set_boundary_target_progress"):
        magnet_model.set_boundary_target_progress(progress)
    else:
        magnet_model.boundary_target_progress = progress


_set_boundary_temperature(model, BOUNDARY_TEMP)
_set_boundary_target_progress(model, BOUNDARY_TARGET_PROGRESS)
# model.set_downsample_gradients_enabled(False)

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

MODEL_NAME = "testing"
MODEL_DIR = Path("./models") / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    # change to a repo name of your choice
    output_dir=str(MODEL_DIR),

    per_device_train_batch_size=16,
    per_device_eval_batch_size=128,

    fp16=True,
    fp16_full_eval=True,
    # bf16=True,
    # bf16_full_eval=True,

    learning_rate=5e-5,
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
    weight_decay=1e-4,

    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    max_grad_norm=2.0,
)


class MagnetSeq2SeqTrainer(Seq2SeqTrainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args)

        # NOTE: differential learning rates are disabled for now. The original
        # implementation is kept below for convenience:
        #
        # boundary_params = []
        # other_params = []
        # for name, param in self.model.named_parameters():
        #     if not param.requires_grad:
        #         continue
        #     if "boundary_predictors" in name:
        #         boundary_params.append(param)
        #     else:
        #         other_params.append(param)
        # base_lr = self.args.learning_rate
        # param_groups = []
        # if other_params:
        #     param_groups.append({"params": other_params, "lr": base_lr * 2.0})
        # if boundary_params:
        #     param_groups.append({"params": boundary_params, "lr": base_lr / 2.0})
        # if not param_groups:
        #     param_groups = [{"params": self.model.parameters()}]

        params = [param for _, param in self.model.named_parameters()
                  if param.requires_grad]
        if not params:
            params = list(self.model.parameters())

        param_groups = [{"params": params, "lr": self.args.learning_rate}]

        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)


trainer = MagnetSeq2SeqTrainer(
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

        boundary_target_progress = getattr(
            model, "boundary_target_progress", None)
        if boundary_target_progress is not None:
            log_payload["train/boundary_target_progress"] = boundary_target_progress

        # downsample_grad_enabled = getattr(
        #     model, "downsample_gradients_enabled", None)
        # if downsample_grad_enabled is not None:
        #     log_payload["train/downsample_gradients_enabled"] = float(
        #         bool(downsample_grad_enabled))

        wandb.log(log_payload)


# Add compression ratio callback
trainer.add_callback(CompressionRatioCallback())


class FreezeNonBoundaryCallback(TrainerCallback):
    def __init__(self, freeze_steps: int):
        self.freeze_steps = freeze_steps
        self._frozen = False
        self._unfrozen = False

    @staticmethod
    def _is_boundary_parameter(name: str) -> bool:
        return "boundary_predictors" in name

    def _freeze(self, model):
        for name, param in model.named_parameters():
            param.requires_grad = self._is_boundary_parameter(name)
        self._frozen = True

    def _unfreeze(self, model):
        for _, param in model.named_parameters():
            param.requires_grad = True
        self._unfrozen = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None or self.freeze_steps <= 0 or self._frozen:
            return
        self._freeze(model)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is None or not self._frozen or self._unfrozen:
            return
        if state.global_step >= self.freeze_steps:
            self._unfreeze(model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is None or not self._frozen or self._unfrozen:
            return
        self._unfreeze(model)


# trainer.add_callback(FreezeNonBoundaryCallback(FREEZE_NON_BOUNDARY_STEPS))


class BoundaryScheduler(TrainerCallback):
    """Kept for reference; scheduling is disabled by commenting out the callback."""

    def __init__(self, start_temp, end_temp, start_progress, end_progress):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.start_progress = start_progress
        self.end_progress = end_progress

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        # Original scheduling logic retained for future experiments:
        # if model is None:
        #     return
        # total_steps = state.max_steps if state and state.max_steps else None
        # if not total_steps or total_steps <= 0:
        #     return
        # progress = min(1.0, state.global_step / total_steps)
        # current_temp = self.start_temp + (self.end_temp - self.start_temp) * progress
        # current_progress = self.start_progress + (self.end_progress - self.start_progress) * progress
        # _set_boundary_temperature(model, current_temp)
        # _set_boundary_target_progress(model, current_progress)
        return


# Example usage (disabled for now):
# trainer.add_callback(
#     BoundaryScheduler(
#         start_temp=BOUNDARY_TEMP_START,
#         end_temp=BOUNDARY_TEMP_END,
#         start_progress=BOUNDARY_TARGET_PROGRESS_START,
#         end_progress=BOUNDARY_TARGET_PROGRESS_END,
#     )
# )


# class DownsampleGradScheduler(TrainerCallback):
#     def __init__(self, warmup_steps: int):
#         self.warmup_steps = warmup_steps
#         self._enabled = warmup_steps <= 0
#
#     def _set_flag(self, model, enabled: bool):
#         if model is None:
#             return
#         if hasattr(model, "set_downsample_gradients_enabled"):
#             model.set_downsample_gradients_enabled(enabled)
#
#     def on_train_begin(self, args, state, control, model=None, **kwargs):
#         if self.warmup_steps <= 0:
#             self._enabled = True
#             self._set_flag(model, True)
#         else:
#             self._enabled = False
#             self._set_flag(model, False)
#
#     def on_step_begin(self, args, state, control, model=None, **kwargs):
#         if model is None or self._enabled:
#             return
#         if state.global_step >= self.warmup_steps:
#             self._set_flag(model, True)
#             self._enabled = True
#
#     def on_train_end(self, args, state, control, model=None, **kwargs):
#         if not self._enabled:
#             self._set_flag(model, True)
#             self._enabled = True


# trainer.add_callback(DownsampleGradScheduler(DOWNSAMPLE_NO_GRAD_STEPS))

# class EvaluateFirstStepCallback(TrainerCallback):
#     def on_step_begin(self, args, state, control, **kwargs):
#         if state.global_step == 1:
#             control.should_evaluate = True

# trainer.add_callback(EvaluateFirstStepCallback())

trainer.train()
# trainer.save_model(f"./models/{MODEL_NAME}")

# model.save_pretrained(f"./models/magnet-phonemes")
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
