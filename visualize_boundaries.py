#!/usr/bin/env python3
"""Visualize boundary predictions from a MagnetWhisper checkpoint."""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.utils.rnn  # noqa: F401  # ensure pad_sequence is available
import torchaudio

from BoundaryPredictor2 import BoundaryPredictor2
from MagnetWhisper import MagnetWhisper
from BoundaryPredictor1 import BoundaryPredictor1
from utils import downsample
from transformers import WhisperProcessor


def _rewrite_boundary_forward(predictor: BoundaryPredictor2) -> None:
    """Patch predictor.forward so it retains the last hard boundary mask."""

    # Store original forward method
    original_forward = predictor.forward

    def _forward(
        self,
        hidden,
        attention_mask: Optional[torch.Tensor] = None,
        target_boundary_counts: Optional[torch.Tensor] = None,
        return_log_probs: bool = False,
        return_unreduced_boundary_loss: bool = False,
        return_confidence: bool = False,
        return_entropy: bool = False,
        rl: bool = False,
    ):
        # Call the original forward method with all parameters
        result = original_forward(
            hidden=hidden,
            attention_mask=attention_mask,
            target_boundary_counts=target_boundary_counts,
            return_log_probs=return_log_probs,
            return_unreduced_boundary_loss=return_unreduced_boundary_loss,
            return_confidence=return_confidence,
            return_entropy=return_entropy,
            rl=rl,
        )

        # Extract the components we need
        pooled = result[0]
        loss = result[1]
        num_boundaries = result[2]
        total_positions = result[3]
        shortened_attention_mask = result[4]

        # We need to reconstruct the hard boundaries from the forward pass
        # Since BoundaryPredictor2 doesn't expose them directly, we need to recompute
        with torch.no_grad():
            normalized_hidden = torch.nn.functional.normalize(hidden, dim=-1)
            q_hidden = self.q_proj_layer(normalized_hidden[:, :-1])
            k_hidden = self.k_proj_layer(normalized_hidden[:, 1:])

            cos_sim = torch.einsum("bld,bld->bl", q_hidden, k_hidden)
            probs = torch.clamp(
                (1 - (cos_sim + self.similarity_bias)) * 0.5, min=0.0, max=1.0)
            probs = torch.nn.functional.pad(probs, (0, 1), value=0.0)

            # Generate hard boundaries using the same logic
            hard_boundaries = (probs > self.threshold).float()

            if attention_mask is not None:
                hard_boundaries = hard_boundaries * attention_mask

                pad_mask = attention_mask == 0
                if pad_mask.any():
                    first_pad_mask = pad_mask & (
                        pad_mask.long().cumsum(dim=1) == 1)
                    last_real_mask = torch.roll(
                        first_pad_mask, shifts=-1, dims=1)
                    last_real_mask[:, -1] = False
                    boundary_mask = last_real_mask.float()
                    hard_boundaries = torch.maximum(
                        hard_boundaries, boundary_mask)

        # Store for visualization
        self.last_hard_boundaries = hard_boundaries.detach().cpu()
        self.last_attention_mask = attention_mask.detach(
        ).cpu() if attention_mask is not None else None
        self.last_loss = loss
        self.last_probs = probs.detach().cpu()

        # Return the full BoundaryPredictor2 format that MagnetWhisperEncoder expects
        return result

    predictor.forward = _forward.__get__(predictor, BoundaryPredictor2)


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
    model_path = Path("./models/checkpoint-10990")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path: Optional[Path] = audio_path.with_name(
        audio_path.stem + "_boundaries.txt")

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        token=os.environ.get("HF_TOKEN"),
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

    predictor: Optional[BoundaryPredictor2] = None
    for module in model.model.encoder.boundary_predictors:
        if isinstance(module, BoundaryPredictor2):
            predictor = module
            break

    if predictor is None:
        raise RuntimeError("No BoundaryPredictor found in the loaded model.")

    _rewrite_boundary_forward(predictor)

    with torch.no_grad():
        print(f"Calling encoder {model.model.encoder.__class__}")
        model.model.encoder(
            input_features=input_features.to(device),
            attention_mask=attention_mask.to(device),
            return_dict=True,
        )

    with torch.no_grad():
        print(f"Calling decoder {model.model.decoder.__class__}")
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
    boundary_probs = predictor.last_probs.squeeze(0)

    # Print all boundary probabilities and positions
    print("\n=== BOUNDARY ANALYSIS ===")

    if attention_mask_recorded is not None:
        valid_mask = attention_mask_recorded > 0.5
        valid_positions = torch.nonzero(valid_mask, as_tuple=True)[0]
        total_valid_positions = int(valid_mask.sum().item())

        print(f"Total valid positions: {total_valid_positions}")
        print(f"Total sequence length: {len(boundary_mask)}")

        valid_probs = boundary_probs[valid_mask]
        valid_boundaries = boundary_mask[valid_mask]
        num_boundaries = int(valid_boundaries.sum().item())

        print(f"Number of boundaries: {num_boundaries}")
        print(
            f"Compression rate: {num_boundaries / total_valid_positions:.4f}")

        print("\nAll valid positions with probabilities:")
        for i, (pos, prob, is_boundary) in enumerate(zip(valid_positions, valid_probs, valid_boundaries)):
            boundary_marker = " [BOUNDARY]" if is_boundary > 0.5 else ""
            print(
                f"Position {pos.item():4d}: prob={prob.item():.6f}{boundary_marker}")

    else:
        total_positions = len(boundary_mask)
        num_boundaries = int(boundary_mask.sum().item())

        print(f"Total positions: {total_positions}")
        print(f"Number of boundaries: {num_boundaries}")
        print(f"Compression rate: {num_boundaries / total_positions:.4f}")

        print("\nAll positions with probabilities:")
        for i, (prob, is_boundary) in enumerate(zip(boundary_probs, boundary_mask)):
            boundary_marker = " [BOUNDARY]" if is_boundary > 0.5 else ""
            print(f"Position {i:4d}: prob={prob.item():.6f}{boundary_marker}")

    print("=== END BOUNDARY ANALYSIS ===\n")

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
