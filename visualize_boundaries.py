#!/usr/bin/env python3
"""Visualize boundary predictions from a MagnetWhisper checkpoint."""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn  # noqa: F401  # ensure pad_sequence is available
import torchaudio

from MagnetWhisper import MagnetWhisper
from BoundaryPredictor2 import BoundaryPredictor2
from utils import downsample
from transformers import WhisperFeatureExtractor


def _rewrite_boundary_forward(predictor: BoundaryPredictor2) -> None:
    """Patch predictor.forward so it retains the last hard boundary mask."""

    def _forward(self, hidden, attention_mask: Optional[torch.Tensor] = None):
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            self.q_proj_layer(F.normalize(hidden[:, :-1])),
            self.k_proj_layer(F.normalize(hidden[:, 1:]))
        )

        probs = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        probs = F.pad(probs, (1, 0), "constant", 1.0)

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
                first_pad_mask = pad_mask & (pad_mask.long().cumsum(dim=1) == 1)
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

        loss = self.calc_loss(num_boundaries_tensor, total_positions_tensor)
        self.last_loss = loss

        self.last_hard_boundaries = hard_boundaries.detach().cpu()
        self.last_attention_mask = attention_mask.detach().cpu() if attention_mask is not None else None

        num_boundaries = num_boundaries_tensor.item()
        total_positions = total_positions_tensor.item()

        return pooled, loss, num_boundaries, total_positions, shortened_attention_mask

    predictor.forward = _forward.__get__(predictor, BoundaryPredictor2)


def _load_audio(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
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

    boundary_indices = torch.nonzero((boundary_mask > 0.5) & valid_mask, as_tuple=True)[0]
    if boundary_indices.numel() == 0:
        return torch.empty(0)

    positions = cumulative_positions[boundary_indices].float() + 1
    return positions * frame_duration


def visualize_boundaries() -> None:
    audio_path = Path("/path/to/audio.wav")
    model_path = "/path/to/magnet/checkpoint"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path: Optional[Path] = None
    feature_extractor_name: Optional[str] = None

    feature_extractor_ref = feature_extractor_name or model_path
    feature_extractor = WhisperFeatureExtractor.from_pretrained(feature_extractor_ref)

    waveform, sample_rate = _load_audio(audio_path, feature_extractor.sampling_rate)
    audio_duration = waveform.numel() / sample_rate

    inputs = feature_extractor(
        waveform.numpy(), sampling_rate=sample_rate, return_attention_mask=True
    )

    input_features = torch.tensor(inputs["input_features"], dtype=torch.float32)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.float32)

    model = MagnetWhisper.from_pretrained(model_path)
    model.eval()
    model.to(device)

    predictor: Optional[BoundaryPredictor2] = None
    for module in model.model.encoder.boundary_predictors:
        if isinstance(module, BoundaryPredictor2):
            predictor = module
            break

    if predictor is None:
        raise RuntimeError("No BoundaryPredictor2 found in the loaded model.")

    _rewrite_boundary_forward(predictor)

    with torch.no_grad():
        model.model.encoder(
            input_features=input_features.to(device),
            attention_mask=attention_mask.to(device),
            return_dict=True,
        )

    boundary_mask = predictor.last_hard_boundaries.squeeze(0)
    attention_mask_recorded = predictor.last_attention_mask.squeeze(0) if predictor.last_attention_mask is not None else None

    boundary_times = _compute_boundary_times(boundary_mask, attention_mask_recorded, audio_duration)

    time_axis = torch.linspace(0, audio_duration, steps=waveform.numel())

    plt.figure(figsize=(14, 4))
    plt.plot(time_axis.numpy(), waveform.numpy(), label="Waveform")

    for t in boundary_times:
        plt.axvline(t.item(), color="red", linestyle="--", alpha=0.6)

    plt.title("Boundary Predictions")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    if output_path is None:
        output_path = audio_path.with_name(audio_path.stem + "_boundaries.png")
    plt.savefig(output_path)
    plt.close()

    readable_times = ", ".join(f"{t.item():.2f}s" for t in boundary_times)
    print(f"Saved boundary visualization to {output_path}")
    if readable_times:
        print(f"Boundary times: {readable_times}")
    else:
        print("No boundaries detected.")


def main() -> None:
    visualize_boundaries()


if __name__ == "__main__":
    main()
