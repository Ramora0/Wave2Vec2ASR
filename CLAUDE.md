# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MagnetWhisper: A research project that adds learnable boundary prediction and sequence compression to OpenAI's Whisper ASR model. The core idea is inserting BoundaryPredictor modules between Whisper encoder layers that learn to downsample (compress) the hidden state sequence at linguistically meaningful boundaries (e.g., phoneme or syllable boundaries), reducing computation while maintaining ASR quality.

## Running

All scripts are standalone Python files designed for GPU (CUDA) training. No build system or package manager is used beyond pip.

```bash
pip install -r requirements.txt

# Supervised training (Seq2SeqTrainer-based)
python whisper.py

# GRPO reinforcement learning training
python whisper_grpo.py

# Run tests
python tests.py              # MagnetAttention padding invariance tests
python test_downsample.py    # Downsample function correctness tests
```

Training scripts expect preprocessed LibriSpeech data on disk (see `SCRATCH_PATH` in `librispeech.py`). Metrics are logged to Weights & Biases.

## Architecture

### Model Stack (class hierarchy via runtime `__class__` reassignment)

```
MagnetWhisper (extends WhisperForConditionalGeneration)
├── MagnetWhisperModel (extends WhisperModel)
│   ├── MagnetWhisperEncoder (extends WhisperEncoder)
│   │   ├── boundary_predictors: ModuleList[BoundaryPredictor | Identity] (12 slots, one per layer)
│   │   ├── additional_convs: Optional extra Conv1d layers for fixed downsampling (ConvConfig)
│   │   └── MagnetEncoderLayer → MagnetAttention (RoPE replaces sinusoidal positional embeddings)
│   └── MagnetWhisperDecoder (extends WhisperDecoder)
│       └── MagnetDecoderLayer → MagnetAttention (self-attn + cross-attn both use RoPE)
```

Models are initialized by loading a pretrained Whisper checkpoint, then converting via `__class__` reassignment and calling `load_magnet()`. This is done in two ways:
- Fresh init: `model.__class__ = MagnetWhisper; model.load_magnet(layer_priors, conv_config=...)`
- From saved: `MagnetWhisper.from_pretrained(path)` (loads `boundary_params.pt`, `boundary_predictors.bin`, `additional_convs.bin`)

### Boundary Prediction Flow

In `MagnetWhisperEncoder.forward()`, for each of the 12 encoder layers:
1. Run boundary predictor (if not Identity) → produces compressed hidden states + boundary loss
2. Run the encoder layer on the (possibly compressed) hidden states

BoundaryPredictor1 (primary): MLP produces per-token boundary probabilities → Gumbel-Sigmoid (training) or threshold (eval) → hard boundaries → attention-based or weighted-mean pooling to compress sequence → sinusoidal positional re-embedding.

### Downsampling

`utils.py` contains the core `downsample()` function: given boundaries (B×L) and hidden states (L×B×D), produces segment-averaged representations (S×B×D). Note the L×B×D (sequence-first) convention for hidden states in downsample. `fast_downsample.py` adds a custom autograd Function with finite-difference gradients for boundary positions.

### Loss Functions (`loss.py`)

- `binomial_loss`: Penalizes deviation from target boundary rate
- `binomial_loss_from_target_counts`: Per-sample loss matching specific boundary counts
- `repulsion_loss`: Conv1d-based penalty for adjacent boundaries

### Training Modes

1. **Supervised** (`whisper.py`): Uses HuggingFace `Seq2SeqTrainer` with callbacks for compression scheduling, temperature annealing, and boundary loss weight scheduling. Target boundary counts come from phoneme counts via `g2p_en`.

2. **GRPO** (`whisper_grpo.py` + `grpo_trainer.py`): Reinforcement learning where K boundary configurations are sampled per audio, rewards = -loss, advantages computed per-group. Gradients flow only through boundary predictor log-probs.

### Data Pipeline (`librispeech.py`)

`DataCollatorSpeechSeq2SeqWithPadding` pads features and computes phoneme-count targets per utterance using `g2p_en`. The collator outputs `target_boundary_counts` shaped as `(1, batch_size)` (note: first dim is number of predictor layers receiving targets).

### Key Conventions

- Hidden dim is 768 (Whisper small)
- All 4 BoundaryPredictor variants (1-4) follow the same interface: `(hidden, attention_mask, target_boundary_counts=..., rl=...) → (pooled, loss, num_boundaries, total_positions, shortened_mask, ...)`
- Attention masks are 1D: `(batch, seq_len)` where 1=valid, 0=padded. `MagnetAttention` takes `query_mask_1d` and `key_mask_1d` separately.
- `MagnetAttention` uses RoPE instead of Whisper's learned positional embeddings. The encoder skips `embed_positions` entirely.
