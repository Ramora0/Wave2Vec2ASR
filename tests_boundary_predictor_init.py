import torch

from BoundaryPredictor1 import BoundaryPredictor1


def analyze_boundary_init(
    input_dim: int = 768,
    hidden_dim: int = 768,
    batch_size: int = 64,
    seq_len: int = 512,
    n_relaxed_samples: int = 512,
    threshold_grid: torch.Tensor | None = None,
    seed: int = 0,
    target_ratio: float = 0.25,
) -> tuple[float, float]:
    """Inspect BoundaryPredictor1's initialization statistics.

    Returns the (threshold, boundary_fraction) pair closest to ``target_ratio``.
    """
    torch.manual_seed(seed)
    predictor = BoundaryPredictor1(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        prior=target_ratio,
        temp=1.0,
        threshold=0.5,
    ).eval()

    hidden = torch.randn(batch_size, seq_len, input_dim)

    with torch.no_grad():
        logits = predictor.boundary_mlp(hidden).squeeze(-1)
        probs = torch.sigmoid(logits)

    print(
        f"logit mean={logits.mean():.4f}, std={logits.std(unbiased=False):.4f}")
    print(
        f"prob  mean={probs.mean():.4f}, std={probs.std(unbiased=False):.4f}")

    relaxed = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
        predictor.temp,
        probs=probs,
    )
    samples = torch.stack([relaxed.rsample()
                          for _ in range(n_relaxed_samples)])

    if threshold_grid is None:
        threshold_grid = torch.linspace(0.05, 0.95, 19)

    best_threshold = float("inf")
    best_ratio = float("inf")
    best_gap = float("inf")

    for threshold in threshold_grid:
        boundary_fraction = (samples > threshold).float().mean().item()
        gap = abs(boundary_fraction - target_ratio)
        print(
            f"threshold={float(threshold):.2f} → boundary fraction={boundary_fraction:.4f}"
        )
        if gap < best_gap:
            best_threshold = float(threshold)
            best_ratio = boundary_fraction
            best_gap = gap

    print(
        f"\n≈{1/target_ratio:.1f}× compression: threshold ≈{best_threshold:.3f} "
        f"(boundary fraction {best_ratio:.3f})"
    )
    return best_threshold, best_ratio


if __name__ == "__main__":
    analyze_boundary_init()
