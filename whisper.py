from MagnetWhisperEncoder import ConvConfig
import torch
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

# Load the model from old-save
# model = MagnetWhisper.from_pretrained("./models/old-save")

# # Load the model
# model = WhisperForConditionalGeneration.from_pretrained(
#     "openai/whisper-small",
#     # torch_dtype=torch.float16,
#     token=os.environ.get("HF_TOKEN")
#     # attn_implementation="flash_attention_2"
# )
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small.en")

# # Convert to the custom MagnetWhisper stack and enable BoundaryPredictor1.
model.__class__ = MagnetWhisper
BOUNDARY_TEMP = 1  # Final temperature we keep fixed during this run
# Max compression, i.e., syllable target throughout training
BOUNDARY_TARGET_PROGRESS = 1.0
# FREEZE_NON_BOUNDARY_STEPS = 250
# DOWNSAMPLE_NO_GRAD_STEPS = 17600
# boundary_priors = [(0, 0.08)]

conv_config = ConvConfig.from_strides([2,2])
model.load_magnet([], conv_config=conv_config)


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
# model = torch.compile(model)
# Don't manually convert to fp16 - let the trainer's AMP handle it
# model.half()  # Convert model to fp16

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

    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,

    # gradient_accumulation_steps=2,
    gradient_checkpointing=True,

    fp16=True,
    fp16_full_eval=True,
    # bf16=True,
    # bf16_full_eval=True,

    learning_rate=1e-4,
    warmup_ratio=0.1,
    # max_steps=16000,
    num_train_epochs=5,
    eval_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=16000,
    save_total_limit=2,
    eval_steps=1000,
    logging_steps=100,
    report_to="wandb",
    greater_is_better=False,
    weight_decay=0.01,

    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    max_grad_norm=2.0,
)


class MagnetSeq2SeqTrainer(Seq2SeqTrainer):
    # def create_optimizer(self):
    #     if self.optimizer is not None:
    #         return

    #     optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
    #         self.args)

    #     boundary_params = []
    #     other_params = []
    #     for name, param in self.model.named_parameters():
    #         if not param.requires_grad:
    #             continue
    #         if "boundary_predictors" in name:
    #             boundary_params.append(param)
    #         else:
    #             other_params.append(param)

    #     # The weight_decay argument in Seq2SeqTrainingArguments is a global setting.
    #     # To apply it only to boundary_params, we create two parameter groups.
    #     weight_decay = self.args.weight_decay
    #     param_groups = [
    #         {"params": other_params, "weight_decay": 0.0},
    #         {"params": boundary_params, "weight_decay": weight_decay},
    #     ]

    #     # The weight_decay is now specified in the param_groups, so remove it from the optimizer_kwargs
    #     # to avoid conflicts if it's present.
    #     if 'weight_decay' in optimizer_kwargs:
    #         del optimizer_kwargs['weight_decay']

    #     self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
    pass


trainer = MagnetSeq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation.clean"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)


class CompressionRatioCallback(TrainerCallback):
    """Callback to log compression ratios to wandb during training"""

    def __init__(self):
        super().__init__()
        self.train_step = 0

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None:
            return

        compression_ratio = model.get_and_reset_compression_ratio()
        boundary_loss = model.get_and_reset_boundary_loss()

        logs["train/compression_ratio"] = compression_ratio

        if boundary_loss is not None:
            logs["train/boundary_loss"] = boundary_loss

        boundary_temp = getattr(model, "boundary_temperature", None)
        if boundary_temp is not None:
            logs["train/boundary_temperature"] = boundary_temp

        boundary_target_progress = getattr(
            model, "boundary_target_progress", None)
        if boundary_target_progress is not None:
            logs["train/boundary_target_progress"] = boundary_target_progress

        # Log boundary CV (coefficient of variation for boundary spacing)
        boundary_cv = getattr(model, "_boundary_cv", None)
        if boundary_cv is not None:
            logs["train/boundary_cv"] = boundary_cv

        # Log boundary adjacent percentage (% of boundaries with neighbor at distance 1)
        boundary_adjacent_pct = getattr(model, "_boundary_adjacent_pct", None)
        if boundary_adjacent_pct is not None:
            logs["train/boundary_adjacent_pct"] = boundary_adjacent_pct

        # Log scheduled_prior and boundary_loss_weight from boundary predictors
        predictors = getattr(model.model.encoder, "boundary_predictors", [])
        if predictors:
            # Log the scheduled prior value
            if hasattr(predictors[0], "get_scheduled_prior"):
                scheduled_prior = predictors[0].get_scheduled_prior()
                logs["train/scheduled_prior"] = scheduled_prior

            # Log the boundary loss weight
            if hasattr(predictors[0], "boundary_loss_weight"):
                loss_weight = predictors[0].boundary_loss_weight
                logs["train/boundary_loss_weight"] = loss_weight

        # downsample_grad_enabled = getattr(
        #     model, "downsample_gradients_enabled", None)
        # if downsample_grad_enabled is not None:
        #     logs["train/downsample_gradients_enabled"] = float(
        #         bool(downsample_grad_enabled))

        wandb.log(logs, step=state.global_step)

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Log metrics after evaluation"""
        if metrics is None:
            return

        # Log compression ratio from evaluation
        compression_ratio = model.get_and_reset_compression_ratio()
        if compression_ratio is not None:
            eval_log = {"eval/compression_ratio": compression_ratio}
            wandb.log(eval_log, step=state.global_step)
            metrics["compression_ratio"] = compression_ratio

        # Log boundary CV from evaluation
        boundary_cv = getattr(model, "_boundary_cv", None)
        if boundary_cv is not None:
            eval_log = {"eval/boundary_cv": boundary_cv}
            wandb.log(eval_log, step=state.global_step)
            # Also add to metrics dict that gets printed
            metrics["boundary_cv"] = boundary_cv

        # Log boundary adjacent percentage from evaluation
        boundary_adjacent_pct = getattr(model, "_boundary_adjacent_pct", None)
        if boundary_adjacent_pct is not None:
            eval_log = {"eval/boundary_adjacent_pct": boundary_adjacent_pct}
            wandb.log(eval_log, step=state.global_step)
            # Also add to metrics dict that gets printed
            metrics["boundary_adjacent_pct"] = boundary_adjacent_pct


# Add compression ratio callback
trainer.add_callback(CompressionRatioCallback())


class CompressionScheduler(TrainerCallback):
    """
    Schedule the compression rate during training.

    The schedule_fn takes training progress (0.0 to 1.0) and returns
    compression_schedule value (0.0 to 1.0) where:
    - 0.0 = no compression (every token is a boundary)
    - 1.0 = max compression (only target_boundary_counts boundaries)
    """

    def __init__(self, schedule_fn=None, start_value=0.0, end_value=1.0):
        """
        Args:
            schedule_fn: Optional function that takes progress (0-1) and returns compression (0-1).
                        If None, uses linear schedule from start_value to end_value.
            start_value: Starting compression value (used if schedule_fn is None)
            end_value: Ending compression value (used if schedule_fn is None)
        """
        self.schedule_fn = schedule_fn
        self.start_value = start_value
        self.end_value = end_value

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        total_steps = state.max_steps if state and state.max_steps else None
        if not total_steps or total_steps <= 0:
            return

        # Calculate training progress (0.0 to 1.0)
        progress = min(1.0, state.global_step / total_steps)

        # Compute compression schedule value
        if self.schedule_fn is not None:
            compression_value = self.schedule_fn(progress)
        else:
            # Linear schedule from start to end
            compression_value = self.start_value + \
                (self.end_value - self.start_value) * progress

        # Set compression schedule for all boundary predictors
        predictors = getattr(model.model.encoder, "boundary_predictors", [])
        for predictor in predictors:
            if hasattr(predictor, "set_compression_schedule"):
                predictor.set_compression_schedule(compression_value)


COMPRESSION_SCHEDULE_STEPS = 12


def compression_schedule(progress):
    """Advance compression in discrete steps during warmup, then hold at 1.0"""
    # steps = 3
    # return int(steps * progress + 1) / steps
    return 1


trainer.add_callback(CompressionScheduler(
    schedule_fn=compression_schedule))


class BoundaryLossWeightScheduler(TrainerCallback):
    """
    Linearly schedule the boundary loss weight from 0 to 1 over training.

    This allows the model to initially train without boundary loss constraints,
    then gradually introduce the boundary loss penalty as training progresses.
    """

    def __init__(self, start_weight=0.0, end_weight=1.0):
        """
        Args:
            start_weight: Initial boundary loss weight (default: 0.0)
            end_weight: Final boundary loss weight (default: 1.0)
        """
        self.start_weight = start_weight
        self.end_weight = end_weight

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        total_steps = state.max_steps if state and state.max_steps else None
        if not total_steps or total_steps <= 0:
            return

        # Calculate training progress (0.0 to 1.0)
        progress = min(1.0, state.global_step / total_steps)

        # Linear interpolation from start_weight to end_weight
        weight = self.start_weight + \
            (self.end_weight - self.start_weight) * progress

        # Set boundary loss weight for all boundary predictors
        predictors = getattr(model.model.encoder, "boundary_predictors", [])
        for predictor in predictors:
            if hasattr(predictor, "set_boundary_loss_weight"):
                predictor.set_boundary_loss_weight(weight)


# trainer.add_callback(BoundaryLossWeightScheduler(
#     start_weight=1.0, end_weight=1.0))


class TemperatureScheduler(TrainerCallback):
    """
    Linearly schedule the temperature from a start to an end value over training.
    """

    def __init__(self, start_temp=1.0, end_temp=0.0):
        self.start_temp = start_temp
        self.end_temp = end_temp

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        total_steps = state.max_steps if state and state.max_steps else None
        if not total_steps or total_steps <= 0:
            return

        progress = min(1.0, state.global_step / total_steps)
        temperature = self.start_temp + \
            (self.end_temp - self.start_temp) * progress

        _set_boundary_temperature(model, temperature)


trainer.add_callback(TemperatureScheduler(start_temp=1, end_temp=0.0))


# class EvaluateFirstStepCallback(TrainerCallback):
#     def on_step_begin(self, args, state, control, **kwargs):
#         if state.global_step == 1:
#             control.should_evaluate = True

# trainer.add_callback(EvaluateFirstStepCallback())

trainer.train()
# trainer.save_model(f"./models/old-save")

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

compression_ratio = model.get_and_reset_compression_ratio()
wandb.log({
    "evaluation/wer": 100 * wer,
    "evaluation/compression_ratio": compression_ratio,
})
