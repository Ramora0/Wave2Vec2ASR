import torch

from old_downsample import downsample as legacy_downsample
from utils import downsample as new_downsample

BATCH_SIZE = 1
SEQ_LEN = 12
MODEL_DIM = 2
BOUNDARY_MODES = ("hard", "random")
RNG_SEED = 1234
TOL = 1e-4


def _generate_boundaries(mode: str) -> torch.Tensor:
    boundaries = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.float32)

    if mode == "hard":
        step = max(1, SEQ_LEN // 4)
        for b in range(BATCH_SIZE):
            idxs = torch.arange(step - 1, SEQ_LEN - 1, step)
            boundaries[b, idxs] = 1.0
    elif mode == "random":
        prob = 0.35
        mask = torch.bernoulli(torch.full_like(boundaries, prob))
        mask[:, 0] = 0
        boundaries = mask
    else:
        raise ValueError(f"Unsupported boundary mode: {mode}")

    # Guarantee at least one boundary per sequence by toggling the last valid
    for b in range(BATCH_SIZE):
        if boundaries[b].sum() == 0:
            boundaries[b, SEQ_LEN - 2] = 1.0

    return boundaries


def _generate_hidden() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, SEQ_LEN, MODEL_DIM, dtype=torch.float32)


def _manual_segment_means(boundaries: torch.Tensor,
                          hidden: torch.Tensor) -> torch.Tensor:
    per_item_segments = boundaries.sum(dim=1, dtype=torch.long)
    max_segments = int(per_item_segments.max().item())

    if max_segments == 0:
        return hidden.new_zeros((0, BATCH_SIZE, MODEL_DIM))

    outputs = hidden.new_zeros((max_segments, BATCH_SIZE, MODEL_DIM))

    for b in range(BATCH_SIZE):
        start = 0
        seg_idx = 0
        for t in range(SEQ_LEN):
            if boundaries[b, t] == 1:
                segment = hidden[b, start:t + 1]
                outputs[seg_idx, b] = segment.mean(dim=0)
                seg_idx += 1
                start = t + 1

    return outputs


def _assert_trailing_zero(name: str,
                          pooled: torch.Tensor,
                          per_item_segments: torch.Tensor,
                          tol: float) -> None:
    for b in range(pooled.size(1)):
        expected = int(per_item_segments[b].item())
        tail = pooled[expected:, b]
        if tail.numel() and tail.abs().max().item() > tol:
            raise RuntimeError(
                f"{name} produced non-zero tail entries\n"
                f"sequence={b}\nexpected_segments={expected}\n"
                f"tail={tail.detach().cpu()}"
            )


def _assert_close(name: str,
                  actual: torch.Tensor,
                  expected: torch.Tensor,
                  tol: float) -> None:
    if not torch.allclose(actual, expected, atol=tol, rtol=0.0):
        raise RuntimeError(
            f"{name} mismatch\n"
            f"actual={actual.detach().cpu()}\n"
            f"expected={expected.detach().cpu()}"
        )


def _run_once(mode: str) -> None:
    boundaries = _generate_boundaries(mode)
    hidden = _generate_hidden()

    manual = _manual_segment_means(boundaries, hidden)
    per_item_segments = boundaries.sum(dim=1, dtype=torch.long)
    max_expected = int(per_item_segments.max().item())

    hidden_time_major = hidden.transpose(0, 1)
    legacy = legacy_downsample(boundaries, hidden_time_major)
    new = new_downsample(boundaries, hidden)

    print("\n=== Mode:", mode, "===")
    print("Boundaries:\n", boundaries)
    print("Hidden (B x L x D):\n", hidden)
    print("Legacy pooled (S x B x D):\n", legacy)
    print("New pooled (S x B x D):\n", new)
    print("Manual means (S x B x D):\n", manual)

    if legacy.size(0) != max_expected:
        raise RuntimeError(
            f"Legacy pooled first dimension {legacy.size(0)}"
            f" disagrees with max boundary count {max_expected}\n"
            f"boundaries={boundaries.detach().cpu()}\n"
            f"legacy_shape={tuple(legacy.shape)}"
        )

    if new.size(0) != max_expected:
        raise RuntimeError(
            f"New pooled first dimension {new.size(0)}"
            f" disagrees with max boundary count {max_expected}\n"
            f"boundaries={boundaries.detach().cpu()}\n"
            f"new_shape={tuple(new.shape)}"
        )

    _assert_trailing_zero("Legacy", legacy, per_item_segments, TOL)
    _assert_trailing_zero("New", new, per_item_segments, TOL)

    if manual.size(0) != max_expected:
        raise RuntimeError(
            f"Manual segment tensor has unexpected shape {manual.size()}"
            f" vs expected {max_expected}\n"
            f"boundaries={boundaries.detach().cpu()}"
        )

    if max_expected > 0:
        _assert_close("Legacy vs manual", legacy, manual, TOL)

    if torch.isnan(new).any() or torch.isnan(legacy).any():
        raise RuntimeError(
            f"NaNs detected in downsample outputs\n"
            f"new={new.detach().cpu()}\nlegacy={legacy.detach().cpu()}"
        )

    print(f"Mode {mode}: OK (segments={max_expected})")


if __name__ == "__main__":
    torch.manual_seed(RNG_SEED)
    for mode in BOUNDARY_MODES:
        _run_once(mode)
