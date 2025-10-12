"""Toy example that inspects gradients through utils.downsample.

Run with `python3 example.py`.
The script creates two obvious acoustic clusters, places the segment
boundaries in the wrong spots, and shows how the gradients point back
toward the correct boundaries.
"""

import torch
import torch.nn.functional as F

from utils import downsample


def format_tensor(t: torch.Tensor) -> str:
    """Round a tensor to 4 decimals for concise printing."""
    return str(torch.round(t * 10000) / 10000)


def main() -> None:
    torch.set_printoptions(precision=4, sci_mode=False)

    # Hidden states form two clear clusters.
    hidden_values = torch.tensor([
        [1.0, 0.2],
        [1.3, -0.1],
        [0.8, 0.3],
        [5.0, 5.2],
        [5.4, 4.9],
        [4.8, 5.1],
    ])
    T, C = hidden_values.shape

    # Downsample expects (T, B, C); we keep batch size B=1 for clarity.
    hidden_wrong = hidden_values.view(T, 1, C).clone().requires_grad_(True)

    # Boundaries are misplaced: both "1"s are an index too early.
    wrong_boundaries = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]], requires_grad=True)

    # Gold-standard boundaries land on the last frame of each cluster.
    correct_boundaries = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]])

    # Target summary uses the correct boundaries but is treated as a constant.
    with torch.no_grad():
        target_summary = downsample(correct_boundaries, hidden_values.view(T, 1, C))

    output = downsample(
        wrong_boundaries,
        hidden_wrong,
    )
    loss = F.mse_loss(output, target_summary)
    loss.backward()

    print("===== Forward =====")
    print("Hidden states (T x C):", format_tensor(hidden_values))
    print("Wrong boundaries:", format_tensor(wrong_boundaries.detach()))
    print("Correct boundaries:", format_tensor(correct_boundaries))
    print("Summary with wrong boundaries:", format_tensor(output.detach().squeeze(1)))
    print("Target summary:", format_tensor(target_summary.squeeze(1)))
    print(f"Loss: {loss.item():.6f}\n")

    print("===== Gradients from backward =====")
    print("Boundary grads:", format_tensor(wrong_boundaries.grad))
    print("Hidden grads:", format_tensor(hidden_wrong.grad.squeeze(1)))

    print(
        "\nInterpretation: The boundary gradient is strongly negative at index 2, "
        "which would raise the boundary value there under gradient descent, "
        "moving it toward the true segment break."
    )



if __name__ == "__main__":
    main()
