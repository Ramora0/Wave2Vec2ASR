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

# Convert to the custom MagnetWhisper stack and enable BoundaryPredictor2 with
# masking-aware priors on every encoder layer.
model.__class__ = MagnetWhisper
boundary_priors = [(1, 0.25)]
model.load_magnet(boundary_priors, "BoundaryPredictor1")

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
    num_train_epochs=1,
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

        wandb.log(log_payload)


# Add compression ratio callback
trainer.add_callback(CompressionRatioCallback())

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
