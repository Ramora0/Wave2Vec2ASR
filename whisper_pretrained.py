import torch
from transformers import TrainerCallback
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
from pathlib import Path
from MagnetWhisper import MagnetWhisper
from transformers import WhisperForConditionalGeneration, WhisperConfig

import wandb

from librispeech import (
    create_librispeech_data_module,
)

print("Starting baseline Whisper training with random initialization (no boundary predictor)...")


# Load only the config (architecture) from checkpoint, then create model with random weights
config = WhisperConfig.from_pretrained(
    "/users/PAS2836/leedavis/research/whisper/models/attention-mask/checkpoint-8789")
model = WhisperForConditionalGeneration(config)

# Convert to the custom MagnetWhisper stack without boundary predictors
model.__class__ = MagnetWhisper
model.load_magnet([], "BoundaryPredictor2")


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


# _set_boundary_temperature(model, BOUNDARY_TEMP)
# _set_boundary_target_progress(model, BOUNDARY_TARGET_PROGRESS)

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

MODEL_NAME = "baseline-whisper-from-scratch"
MODEL_DIR = Path("./models") / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(MODEL_DIR),

    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,

    # Temporarily disabled to debug backward graph error
    gradient_checkpointing=True,
    use_reentrant_gradients=False,

    fp16=True,
    fp16_full_eval=True,

    # Pretraining requires MUCH more training than finetuning
    # LibriSpeech is ~1000 hours vs Whisper's original 680k hours
    # Increased from 5 to 30 epochs to give random init a fair chance
    num_train_epochs=15,

    # Lower initial LR for more stable training from random init
    # Random weights need gentler optimization initially
    learning_rate=1e-4,

    # Longer warmup (20% vs 10%) helps stabilize early training
    # Critical when starting from random weights
    warmup_ratio=0.2,

    eval_strategy="steps",
    # Temporarily disabled to debug backward graph error
    predict_with_generate=False,
    generation_max_length=225,

    # Save more frequently to not lose progress if training crashes
    save_steps=4000,
    save_total_limit=3,

    # Eval more frequently to catch early issues
    eval_steps=3000,
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
    pass


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

        boundary_cv = getattr(model, "_boundary_cv", None)
        if boundary_cv is not None:
            logs["train/boundary_cv"] = boundary_cv

        boundary_adjacent_pct = getattr(model, "_boundary_adjacent_pct", None)
        if boundary_adjacent_pct is not None:
            logs["train/boundary_adjacent_pct"] = boundary_adjacent_pct

        predictors = getattr(model.model.encoder, "boundary_predictors", [])
        if predictors:
            if hasattr(predictors[0], "get_scheduled_prior"):
                scheduled_prior = predictors[0].get_scheduled_prior()
                logs["train/scheduled_prior"] = scheduled_prior

            if hasattr(predictors[0], "boundary_loss_weight"):
                loss_weight = predictors[0].boundary_loss_weight
                logs["train/boundary_loss_weight"] = loss_weight

        wandb.log(logs, step=state.global_step)

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Log metrics after evaluation"""
        if metrics is None:
            return

        boundary_cv = getattr(model, "_boundary_cv", None)
        if boundary_cv is not None:
            eval_log = {"eval/boundary_cv": boundary_cv}
            wandb.log(eval_log, step=state.global_step)
            metrics["boundary_cv"] = boundary_cv

        boundary_adjacent_pct = getattr(model, "_boundary_adjacent_pct", None)
        if boundary_adjacent_pct is not None:
            eval_log = {"eval/boundary_adjacent_pct": boundary_adjacent_pct}
            wandb.log(eval_log, step=state.global_step)
            metrics["boundary_adjacent_pct"] = boundary_adjacent_pct


# COMMENTED OUT - CompressionRatioCallback only needed for MagnetWhisper experiments
# trainer.add_callback(CompressionRatioCallback())


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

        progress = min(1.0, state.global_step / total_steps)

        if self.schedule_fn is not None:
            compression_value = self.schedule_fn(progress)
        else:
            compression_value = self.start_value + \
                (self.end_value - self.start_value) * progress

        predictors = getattr(model.model.encoder, "boundary_predictors", [])
        for predictor in predictors:
            if hasattr(predictor, "set_compression_schedule"):
                predictor.set_compression_schedule(compression_value)


def compression_schedule(progress):
    """Advance compression in discrete steps during warmup, then hold at 1.0"""
    return 1


# COMMENTED OUT - These callbacks are only needed for boundary predictor experiments
# trainer.add_callback(CompressionScheduler(
#     schedule_fn=compression_schedule))


# class TemperatureScheduler(TrainerCallback):
#     """
#     Linearly schedule the temperature from a start to an end value over training.
#     """

#     def __init__(self, start_temp=1.0, end_temp=0.0):
#         self.start_temp = start_temp
#         self.end_temp = end_temp

#     def on_step_begin(self, args, state, control, model=None, **kwargs):
#         if model is None:
#             return

#         total_steps = state.max_steps if state and state.max_steps else None
#         if not total_steps or total_steps <= 0:
#             return

#         progress = min(1.0, state.global_step / total_steps)
#         temperature = self.start_temp + \
#             (self.end_temp - self.start_temp) * progress

#         _set_boundary_temperature(model, temperature)


# trainer.add_callback(TemperatureScheduler(start_temp=1, end_temp=0.0))

trainer.train()
