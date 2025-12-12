from transformers import WhisperForConditionalGeneration as WhisperPretrained
import torch
from transformers import TrainerCallback
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
from pathlib import Path
from MagnetWhisper import MagnetWhisper
from transformers import WhisperForConditionalGeneration, WhisperConfig, WhisperTokenizer

import wandb
from transformers import GenerationConfig

from librispeech import (
    create_librispeech_data_module,
)

print("Starting baseline Whisper training with random initialization (no boundary predictor)...")


# Check if we should resume from a checkpoint
resume_checkpoint = "models/whisper-pretrain/checkpoint-8800"
resume_from_checkpoint = Path(
    resume_checkpoint).exists() if resume_checkpoint else False

if resume_from_checkpoint:
    print(f"Resuming training from checkpoint: {resume_checkpoint}")
    # Load the full model with weights from checkpoint
    model = WhisperForConditionalGeneration.from_pretrained(resume_checkpoint)
    generation_config = GenerationConfig.from_pretrained(resume_checkpoint)
    model.generation_config = generation_config

    # Convert to the custom MagnetWhisper stack without boundary predictor
    model.__class__ = MagnetWhisper
    model.load_magnet(predictor_type="none")
else:
    print("Starting training with random initialization...")
    # Load only the config (architecture) from checkpoint, then create model with random weights
    checkpoint_path = "whisper-pretrain/checkpoint-8800"
    config = WhisperConfig.from_pretrained(checkpoint_path)
    model = WhisperForConditionalGeneration(config)
    generation_config = GenerationConfig.from_pretrained(checkpoint_path)
    model.generation_config = generation_config

    # Convert to the custom MagnetWhisper stack without boundary predictor
    model.__class__ = MagnetWhisper
    model.load_magnet(predictor_type="none")
    # model.load_magnet(0.075, predictor_type="BoundaryPredictor2")


def _set_boundary_temperature(magnet_model, temperature):
    if hasattr(magnet_model.model.encoder, "boundary_predictor"):
        predictor = magnet_model.model.encoder.boundary_predictor
        if hasattr(predictor, "temp"):
            predictor.temp = temperature
    magnet_model.boundary_temperature = temperature


def _set_boundary_target_progress(magnet_model, progress):
    if hasattr(magnet_model, "set_boundary_target_progress"):
        magnet_model.set_boundary_target_progress(progress)
    else:
        magnet_model.boundary_target_progress = progress


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

os.environ["WANDB_PROJECT"] = "glimpse-pretraining"

MODEL_NAME = "whisper-pretrain"
MODEL_DIR = Path("./models") / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(MODEL_DIR),

    per_device_train_batch_size=32,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=32,

    # Temporarily disabled to debug backward graph error
    # gradient_checkpointing=True,
    # use_reentrant_gradients=False,

    bf16=True,
    bf16_full_eval=True,

    # Pretraining requires MUCH more training than finetuning
    # LibriSpeech is ~1000 hours vs Whisper's original 680k hours
    # Increased from 5 to 30 epochs to give random init a fair chance
    num_train_epochs=15,

    # Lower initial LR for more stable training from random init
    # Random weights need gentler optimization initially
    learning_rate=2e-4,

    # Longer warmup (20% vs 10%) helps stabilize early training
    # Critical when starting from random weights
    warmup_ratio=0.2,
    lr_scheduler_type="cosine",

    eval_strategy="steps",
    # Temporarily disabled to debug backward graph error
    predict_with_generate=True,
    generation_max_length=225,

    # Save more frequently to not lose progress if training crashes
    save_steps=1100,
    save_total_limit=3,

    # Eval more frequently to catch early issues
    # eval_steps=550,
    eval_steps=1100,
    logging_steps=100,

    report_to="wandb",
    greater_is_better=False,
    weight_decay=0.05,

    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    # dataloader_prefetch_factor=2,
    remove_unused_columns=False,
    max_grad_norm=1.0,
    torch_compile=True,
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

        logs["boundaries/compression_ratio"] = compression_ratio

        if boundary_loss is not None:
            logs["boundaries/boundary_loss"] = boundary_loss

            # Subtract boundary loss from train/loss to show only ASR loss
            if "loss" in logs:
                # Override with ASR-only loss
                logs["loss"] = logs["loss"] - boundary_loss
            if "train/loss" in logs:
                logs["train/loss"] = logs["train/loss"] - \
                    boundary_loss  # Override with ASR-only loss

        boundary_temp = getattr(model, "boundary_temperature", None)
        if boundary_temp is not None:
            logs["boundaries/boundary_temperature"] = boundary_temp

        boundary_target_progress = getattr(
            model, "boundary_target_progress", None)
        if boundary_target_progress is not None:
            logs["boundaries/boundary_target_progress"] = boundary_target_progress

        boundary_cv = getattr(model, "_boundary_cv", None)
        if boundary_cv is not None:
            logs["boundaries/boundary_cv"] = boundary_cv

        boundary_adjacent_pct = getattr(model, "_boundary_adjacent_pct", None)
        if boundary_adjacent_pct is not None:
            logs["boundaries/boundary_adjacent_pct"] = boundary_adjacent_pct

        if hasattr(model.model.encoder, "boundary_predictor"):
            predictor = model.model.encoder.boundary_predictor
            if hasattr(predictor, "get_scheduled_prior"):
                scheduled_prior = predictor.get_scheduled_prior()
                logs["boundaries/scheduled_prior"] = scheduled_prior

        wandb.log(logs, step=state.global_step)

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Log metrics after evaluation"""
        if metrics is None:
            return

        boundary_cv = getattr(model, "_boundary_cv", None)
        if boundary_cv is not None:
            eval_log = {"eval/boundary_cv": boundary_cv}
            wandb.log(eval_log, step=state.global_step)
            metrics["eval/boundary_cv"] = boundary_cv

        boundary_adjacent_pct = getattr(model, "_boundary_adjacent_pct", None)
        if boundary_adjacent_pct is not None:
            eval_log = {"eval/boundary_adjacent_pct": boundary_adjacent_pct}
            wandb.log(eval_log, step=state.global_step)
            metrics["eval/boundary_adjacent_pct"] = boundary_adjacent_pct


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


# trainer.add_callback(CompressionRatioCallback())
# trainer.add_callback(TemperatureScheduler(start_temp=1, end_temp=0.0))

# Resume training from checkpoint if it exists
if resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=resume_checkpoint)
else:
    trainer.train()
