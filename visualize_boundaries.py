#!/usr/bin/env python3
"""Visualize boundary predictions from a MagnetWhisper checkpoint."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.utils.rnn  # noqa: F401  # ensure pad_sequence is available
import torchaudio

from MagnetWhisper import MagnetWhisper
from BoundaryPredictor1 import BoundaryPredictor1
from utils import downsample
from transformers import WhisperProcessor


def _rewrite_boundary_forward(predictor: BoundaryPredictor1) -> None:
    """Patch predictor.forward so it retains the last hard boundary mask."""

    def _forward(
        self,
        hidden,
        attention_mask: Optional[torch.Tensor] = None,
        target_boundary_counts: Optional[torch.Tensor] = None,
    ):
        logits = self.boundary_mlp(hidden).squeeze(-1)
        probs = torch.sigmoid(logits)

        bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=self.temp,
            probs=probs,
        )

        soft_boundaries = bernoulli.rsample()

        hard_boundaries = (soft_boundaries > self.threshold).float()
        hard_boundaries = (
            hard_boundaries - soft_boundaries.detach() + soft_boundaries
        )

        if attention_mask is not None:
            hard_boundaries = hard_boundaries * attention_mask

            pad_mask = attention_mask == 0
            if pad_mask.any():
                first_pad_mask = pad_mask & (
                    pad_mask.long().cumsum(dim=1) == 1)
                last_real_mask = torch.roll(first_pad_mask, shifts=-1, dims=1)
                last_real_mask[:, -1] = False
                hard_boundaries = torch.maximum(
                    hard_boundaries, last_real_mask.float()
                )

        pooled = downsample(hard_boundaries, hidden)
        pooled = pooled.transpose(0, 1)

        shortened_attention_mask = None
        if attention_mask is not None:
            keep_mask = hard_boundaries == 1
            batch_size = attention_mask.shape[0]
            shortened_masks = []

            for b in range(batch_size):
                keep_indices = keep_mask[b].nonzero(as_tuple=True)[0]
                original_mask = attention_mask[b]
                shortened_mask = original_mask[keep_indices]
                shortened_masks.append(shortened_mask)

            shortened_attention_mask = torch.nn.utils.rnn.pad_sequence(
                shortened_masks, batch_first=True, padding_value=0.0)

        num_boundaries_tensor = hard_boundaries.sum()
        if attention_mask is not None:
            total_positions_tensor = attention_mask.sum()
        else:
            total_positions_tensor = torch.tensor(
                hard_boundaries.numel(), device=hard_boundaries.device, dtype=torch.float)

        if target_boundary_counts is not None:
            loss = self.calc_loss_target_counts(
                hard_boundaries,
                attention_mask,
                target_boundary_counts,
            )
        else:
            loss = num_boundaries_tensor.new_tensor(0.0)

        self.last_loss = loss

        self.last_hard_boundaries = hard_boundaries.detach().cpu()
        self.last_attention_mask = attention_mask.detach(
        ).cpu() if attention_mask is not None else None

        num_boundaries = num_boundaries_tensor.item()
        total_positions = total_positions_tensor.item()

        return pooled, loss, num_boundaries, total_positions, shortened_attention_mask

    predictor.forward = _forward.__get__(predictor, BoundaryPredictor1)


def _load_audio(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except RuntimeError:
        print("No torchaudio :(")
        try:
            import librosa

            array, sample_rate = librosa.load(str(path), sr=None, mono=False)
            if array.ndim == 1:
                waveform = torch.from_numpy(array).unsqueeze(0)
            else:
                waveform = torch.from_numpy(array)
        except Exception as librosa_exc:
            raise RuntimeError(
                f"Unable to load audio file {path}. Ensure ffmpeg support is available."
            ) from librosa_exc
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)

    waveform = waveform.to(torch.float32)
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr

    return waveform.squeeze(0), sample_rate


def _compute_boundary_times(boundary_mask: torch.Tensor,
                            attention_mask: Optional[torch.Tensor],
                            audio_duration: float) -> torch.Tensor:
    if boundary_mask.dim() != 1:
        raise ValueError("Expected boundary mask for a single example.")

    if attention_mask is None:
        num_positions = boundary_mask.numel()
        frame_duration = audio_duration / num_positions
        indices = torch.nonzero(boundary_mask > 0.5, as_tuple=True)[0]
        return (indices + 1).float() * frame_duration

    valid_mask = attention_mask > 0.5
    num_valid = int(valid_mask.sum().item())
    if num_valid == 0:
        return torch.empty(0)

    cumulative_positions = valid_mask.cumsum(dim=0) - 1
    cumulative_positions[~valid_mask] = -1

    frame_duration = audio_duration / num_valid

    boundary_indices = torch.nonzero(
        (boundary_mask > 0.5) & valid_mask, as_tuple=True)[0]
    if boundary_indices.numel() == 0:
        return torch.empty(0)

    positions = cumulative_positions[boundary_indices].float() + 1
    return positions * frame_duration


def visualize_boundaries() -> None:
    audio_path = Path("./data/validation-1.mp3")
    model_path = Path("./models/magnet-phonemes")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path: Optional[Path] = audio_path.with_name(
        audio_path.stem + "_boundaries.txt")

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        token="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",
        language="English",
        task="transcribe",
    )

    waveform, sample_rate = _load_audio(
        audio_path, processor.feature_extractor.sampling_rate)
    audio_duration = waveform.numel() / sample_rate

    inputs = processor.feature_extractor(
        waveform.numpy(), sampling_rate=sample_rate, return_attention_mask=True
    )

    input_features = torch.tensor(
        inputs["input_features"], dtype=torch.float32)
    attention_mask = torch.tensor(
        inputs["attention_mask"], dtype=torch.float32)

    model = MagnetWhisper.from_pretrained(str(model_path))
    model.eval()
    model.to(device)

    predictor: Optional[BoundaryPredictor1] = None
    for module in model.model.encoder.boundary_predictors:
        if isinstance(module, BoundaryPredictor1):
            predictor = module
            break

    if predictor is None:
        raise RuntimeError("No BoundaryPredictor found in the loaded model.")

    _rewrite_boundary_forward(predictor)

    with torch.no_grad():
        model.model.encoder(
            input_features=input_features.to(device),
            attention_mask=attention_mask.to(device),
            return_dict=True,
        )

    with torch.no_grad():
        generated_ids = model.generate(
            inputs=input_features.to(device),
            attention_mask=attention_mask.to(device),
        )[0]

    prediction_text = processor.decode(
        generated_ids.cpu(), skip_special_tokens=True
    ).strip()

    boundary_mask = predictor.last_hard_boundaries.squeeze(0)
    attention_mask_recorded = predictor.last_attention_mask.squeeze(
        0) if predictor.last_attention_mask is not None else None

    boundary_times = _compute_boundary_times(
        boundary_mask, attention_mask_recorded, audio_duration)

    boundary_times_list = boundary_times.tolist()
    label_lines = [
        f"{timestamp:.6f}\t{timestamp:.6f}\tboundary_{idx}"
        for idx, timestamp in enumerate(boundary_times_list, start=1)
    ]

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(label_lines))

    if label_lines:
        print(f"Wrote {len(label_lines)} labels to {output_path}")
    else:
        print("No boundaries detected; label file left empty.")

    print("\nModel transcription:")
    print(prediction_text if prediction_text else "(empty)")


def main() -> None:
    visualize_boundaries()


if __name__ == "__main__":
    main()
