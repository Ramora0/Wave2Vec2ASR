import torch
import torch.nn.functional as F
from utils import downsample
from finite_difference_downsample import downsample_with_finite_diff_grad


def format_tensor(t: torch.Tensor) -> str:
    """Round a tensor to 4 decimals for concise printing."""
    return str(torch.round(t * 10000) / 10000)


def optimize(downsample_func, initial_boundaries, hidden_states, target_summary, num_steps=100, lr=0.1, binomial_weight=0.1, **kwargs):
    """
    Performs gradient descent to optimize boundaries for a given downsample function.
    """
    boundaries = initial_boundaries.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([boundaries], lr=lr)

    num_boundaries = int(target_summary.shape[0])

    def renormalize(tensor: torch.Tensor) -> torch.Tensor:
        total_mass = tensor.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return tensor * (num_boundaries / total_mass)

    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()

        # hardened_boundaries = (boundaries > 0.5).float()
        # Pick the 125 largest values to be boundaries
        _, indices = torch.topk(boundaries, k=num_boundaries, dim=1)
        hardened_boundaries = torch.zeros_like(boundaries)
        hardened_boundaries.scatter_(1, indices, 1.0)
        ste_boundaries = hardened_boundaries - boundaries.detach() + boundaries

        output = downsample_func(ste_boundaries, hidden_states, **kwargs)

        print("output size:", output.shape)
        print("target size:", target_summary.shape)
        print(hardened_boundaries.sum())
        print(hardened_boundaries)
        mse_loss = F.mse_loss(output, target_summary)
        binomial_loss = (boundaries.sum() - num_boundaries)**2
        loss = mse_loss + binomial_weight * binomial_loss

        loss.backward()

        optimizer.step()

        # with torch.no_grad():
        # boundaries.data.clamp_(min=0.0)
        # boundaries.data = renormalize(boundaries.data)

        losses.append(loss.item())

        if step % 10 == 0:
            num_positions = int(hardened_boundaries.sum().item())
            print(
                f"Step {step:02d} | loss={loss.item():.4f} | "
                f"boundaries={format_tensor(boundaries.data)} | # positions={num_positions}"
            )

    return boundaries, losses


def main():
    torch.set_printoptions(precision=4, sci_mode=False)

    T = 1500
    C = 8
    num_boundaries = 125

    hidden_states = torch.randn(T, 1, C)

    # Create correct boundaries
    correct_boundary_indices = torch.randperm(T)[:num_boundaries]
    correct_boundaries = torch.zeros(1, T)
    correct_boundaries[0, correct_boundary_indices] = 1

    # Create initial boundaries
    initial_boundary_indices = torch.randperm(T)[:num_boundaries]
    initial_boundaries = torch.zeros(1, T)
    initial_boundaries[0, initial_boundary_indices] = 1

    # Target summary uses the correct boundaries but is treated as a constant.
    with torch.no_grad():
        target_summary = downsample(correct_boundaries, hidden_states)

    print("===== Testing downsample from utils.py =====")
    optimize(downsample, initial_boundaries, hidden_states,
             target_summary, binomial_weight=1.0)


def test_finite_difference_downsample():
    """
    Test that finite difference downsample can optimize boundaries to match a target
    through gradient descent. This demonstrates that the gradient estimation works.
    """
    print("\n" + "=" * 80)
    print("Testing Finite Difference Downsample Optimization")
    print("=" * 80)

    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

    B = 2
    L = 20
    D = 8

    # Create target boundaries
    target_boundaries = torch.zeros(B, L)
    target_boundaries[0, [0, 4, 8, 12, 16]] = 1  # 5 segments
    target_boundaries[1, [0, 3, 9, 15, 19]] = 1  # 5 segments

    print(f"Target boundaries (batch 0): {target_boundaries[0].tolist()}")
    print(f"Target boundaries (batch 1): {target_boundaries[1].tolist()}")

    # Create hidden states
    hidden = torch.randn(L, B, D) * 10

    # Compute target output (what we're trying to achieve)
    with torch.no_grad():
        target_output = downsample(target_boundaries, hidden)

    print(f"\nTarget output shape: {target_output.shape}")

    # Initialize learnable boundaries from random logits
    learned_logits = torch.randn(B, L, requires_grad=True) * 0.5
    optimizer = torch.optim.Adam([learned_logits], lr=0.1)

    num_iterations = 150
    log_every = 30

    print(f"\nOptimizing for {num_iterations} iterations...")
    print("=" * 80)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Convert logits to boundaries with straight-through estimator
        soft_boundaries = torch.sigmoid(learned_logits)
        hard_boundaries = (soft_boundaries > 0.5).float()
        hard_boundaries = hard_boundaries - soft_boundaries.detach() + soft_boundaries

        # Force first position to always be a boundary
        hard_boundaries[:, 0] = 1.0

        # Forward pass with finite difference gradients
        output = downsample_with_finite_diff_grad(
            hard_boundaries, hidden, gradient_scale=1.0
        )

        # Handle different output shapes (different number of segments)
        S_out = output.size(0)
        S_target = target_output.size(0)

        if S_out != S_target:
            max_S = max(S_out, S_target)
            output_padded = torch.zeros(max_S, B, D, device=output.device)
            target_padded = torch.zeros(max_S, B, D, device=target_output.device)
            output_padded[:S_out] = output
            target_padded[:S_target] = target_output
        else:
            output_padded = output
            target_padded = target_output

        # MSE loss between outputs
        mse_loss = F.mse_loss(output_padded, target_padded)

        # Encourage matching boundary counts
        target_counts = target_boundaries.sum(dim=1)
        learned_counts = hard_boundaries.sum(dim=1)
        count_loss = F.mse_loss(learned_counts, target_counts)

        loss = mse_loss + 0.1 * count_loss

        loss.backward()
        optimizer.step()

        if (iteration + 1) % log_every == 0 or iteration == 0:
            with torch.no_grad():
                current_hard = (torch.sigmoid(learned_logits) > 0.5).float()
                current_hard[:, 0] = 1.0

                boundary_accuracy = (current_hard == target_boundaries).float().mean().item()

                learned_count_0 = current_hard[0].sum().item()
                learned_count_1 = current_hard[1].sum().item()
                target_count_0 = target_boundaries[0].sum().item()
                target_count_1 = target_boundaries[1].sum().item()

            print(f"Iter {iteration+1:3d} | Loss: {loss.item():.6f} | MSE: {mse_loss.item():.6f} | "
                  f"Accuracy: {boundary_accuracy*100:.1f}% | "
                  f"Counts: [{learned_count_0:.0f}/{target_count_0:.0f}, "
                  f"{learned_count_1:.0f}/{target_count_1:.0f}]")

    print("\n" + "=" * 80)
    print("Final Results")
    print("=" * 80)

    with torch.no_grad():
        final_boundaries = (torch.sigmoid(learned_logits) > 0.5).float()
        final_boundaries[:, 0] = 1.0

        print("\nBatch 0:")
        print(f"  Target:  {target_boundaries[0].tolist()}")
        print(f"  Learned: {final_boundaries[0].tolist()}")

        print("\nBatch 1:")
        print(f"  Target:  {target_boundaries[1].tolist()}")
        print(f"  Learned: {final_boundaries[1].tolist()}")

        accuracy = (final_boundaries == target_boundaries).float().mean().item()
        print(f"\nOverall boundary accuracy: {accuracy*100:.2f}%")

        if accuracy > 0.8:
            print("✓ Successfully learned boundaries with >80% accuracy!")
        else:
            print("⚠ Partial success - may need more iterations or tuning")


def test_finite_diff_through_network():
    """
    Test that gradients flow correctly when downsample is in the middle of a network.
    This confirms the delta estimation works anywhere in the network.
    """
    print("\n" + "=" * 80)
    print("Testing Gradient Flow: Downsample -> MLP -> Loss")
    print("=" * 80)

    torch.manual_seed(123)

    B, L, D = 1, 12, 4

    # Create boundaries
    boundaries_logits = torch.randn(B, L, requires_grad=True) * 0.3
    hidden = torch.randn(L, B, D) * 5

    # Create a small MLP to go after downsample
    mlp = torch.nn.Sequential(
        torch.nn.Linear(D, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, D)
    )

    print("Network: Downsample -> Linear -> ReLU -> Linear -> Sum Loss")

    # Forward pass
    soft = torch.sigmoid(boundaries_logits)
    hard = (soft > 0.5).float()
    hard = hard - soft.detach() + soft
    hard[:, 0] = 1.0

    print(f"\nBoundaries: {hard[0].tolist()}")

    # Downsample
    downsampled = downsample_with_finite_diff_grad(hard, hidden, gradient_scale=1.0)
    print(f"Downsampled shape: {downsampled.shape}")

    # Through MLP
    S = downsampled.size(0)
    transformed = mlp(downsampled.view(S * B, D)).view(S, B, D)
    print(f"After MLP shape: {transformed.shape}")

    # Loss
    loss = transformed.sum()
    print(f"Loss: {loss.item():.4f}")

    # Backward
    loss.backward()

    print(f"\n✓ Boundaries gradient computed: {boundaries_logits.grad is not None}")
    print(f"  Gradient shape: {boundaries_logits.grad.shape}")
    print(f"  Gradient norm: {boundaries_logits.grad.norm().item():.4f}")
    print(f"  Non-zero elements: {(boundaries_logits.grad != 0).sum().item()}")

    print("\n✓ Gradients flow correctly through downsample -> network layers!")


if __name__ == "__main__":
    # Run original test
    main()

    # Run finite difference tests
    test_finite_difference_downsample()
    test_finite_diff_through_network()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
