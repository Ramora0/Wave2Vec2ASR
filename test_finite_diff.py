import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import downsample
from loss import binomial_loss as boundarypredictor_binomial_loss

# --- Flag to switch between implementations ---
USE_FAST_VERSION = True
GRAD_ZERO_THRESHOLD = 2.0

if USE_FAST_VERSION:
    print("--- Using FAST analytical gradient implementation ---")
    from fast_downsample import (
        downsample_with_fast_finite_diff_grad as downsample_with_grad,
        get_last_fast_downsample_debug,
    )
else:
    print("--- Using SLOW reference gradient implementation ---")
    from finite_difference_downsample import downsample_with_finite_diff_grad as downsample_with_grad

    def get_last_fast_downsample_debug():
        return None


def test_finite_difference_downsample(
    local_similarity_window=50,
    compression_rate=1.0 / 12.0,
    sequence_length=24,
):
    """
    Test that finite difference downsample can optimize boundaries to match a target
    through gradient descent. Uses realistic scale with a configurable sequence length,
    compression rate (defaults to ~1/12 boundaries per timestep).
    """
    print("=" * 80)
    print(
        f"Testing Finite Difference Downsample Optimization (Local Similarity: {local_similarity_window})")
    print("=" * 80)

    torch.manual_seed(0)
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

    B = 1  # Single batch for simplicity
    L = int(sequence_length)
    if L <= 0:
        raise ValueError("sequence_length must be a positive integer.")
    D = 8  # Hidden dimension
    if compression_rate <= 0 or compression_rate > 1:
        raise ValueError("compression_rate must be in the range (0, 1].")
    K = max(1, int(math.ceil(L * compression_rate)))  # Number of boundaries

    print(f"Sequence length: {L}, Number of boundaries: {K}, Hidden dim: {D}")
    print(
        f"Compression rate: {compression_rate:.6f} (target boundaries / total)")

    # Create hidden states with local similarity
    print(f"Local similarity window: {local_similarity_window}")
    low_res_L = L // local_similarity_window
    if low_res_L < 2:
        low_res_L = 2
    low_res_hidden = torch.randn(low_res_L, B, D) * 10
    hidden = F.interpolate(
        low_res_hidden.permute(1, 2, 0),
        size=L,
        mode='linear',
        align_corners=False
    ).permute(2, 0, 1)

    if K < 1:
        raise ValueError("Number of boundaries K must be at least 1.")

    print("\nHidden states (per timestep batch=0):")
    hidden_list = hidden[:, 0, :].tolist()
    for idx, vec in enumerate(hidden_list):
        formatted = ' '.join(f"{val:.4f}" for val in vec)
        print(f"{idx:02d}: {formatted}")

    if K > 1 and L > 1:
        random_indices = torch.randperm(L - 1)[:K - 1]
    else:
        random_indices = torch.empty(0, dtype=torch.long)
    final_index = torch.tensor([L - 1], dtype=torch.long)
    target_boundary_indices = torch.cat(
        [random_indices, final_index]).sort()[0]
    target_boundaries = torch.zeros(B, L)
    target_boundaries[0, target_boundary_indices] = 1.0

    target_idx_set = set(target_boundary_indices.tolist())
    ascii_threshold = 128
    target_ascii = ''.join('|' if idx in target_idx_set else ' ' for idx in range(L)) \
        if L <= ascii_threshold else None

    print(
        f"\nTarget boundary indices (first 10): {target_boundary_indices[:10].tolist()}")
    print(f"Target boundary count: {target_boundaries.sum().item():.0f}")

    # Compute target output (what we're trying to achieve)
    with torch.no_grad():
        target_output = downsample(target_boundaries, hidden)

    print(f"Target output shape: {target_output.shape}")

    print("\nTarget pooled hidden states:")
    target_pooled = target_output[:, 0, :].tolist()
    for idx, vec in enumerate(target_pooled):
        formatted = ' '.join(f"{val:.4f}" for val in vec)
        print(f"{idx:02d}: {formatted}")

    # Initialize learnable logits to encourage an average matching compression_rate
    binomial_target_rate = compression_rate
    initial_logits = torch.log(torch.tensor(
        binomial_target_rate / (1 - binomial_target_rate)))
    learned_logits = torch.full(
        (B, L), initial_logits.item()) + torch.randn(B, L) * 0.01
    learned_logits.requires_grad = True
    optimizer = torch.optim.Adam([learned_logits], lr=0.05)

    # Temperature for RelaxedBernoulli
    temperature = 1

    num_iterations = 5000
    log_every = 10

    print(
        f"\nOptimizing for {num_iterations} iterations with RelaxedBernoulli sampling...")
    print("=" * 80)

    progress_bar = tqdm(range(num_iterations), leave=False)

    grad_marker_str = None

    final_ste_boundaries = None

    for iteration in progress_bar:
        optimizer.zero_grad()

        # Use RelaxedBernoulli for differentiable boundary sampling
        probs = torch.sigmoid(learned_logits)
        # bernoulli_dist = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
        #     temperature=temperature,
        #     probs=probs
        # )
        # soft_boundaries = bernoulli_dist.rsample()

        # Create hard boundaries with a straight-through estimator
        # hard_boundaries = (soft_boundaries > 0.5).float()
        # Pick top K-1 boundaries (excluding the last position which is always a boundary)
        hard_boundaries = torch.zeros_like(learned_logits)
        if K > 1:
            _, topk_indices = torch.topk(
                learned_logits[:, :-1], k=K - 1, dim=1, sorted=False)
            hard_boundaries.scatter_(1, topk_indices, 1.0)
        hard_boundaries[:, -1] = 1.0
        ste_boundaries = hard_boundaries - learned_logits.detach() + learned_logits

        if iteration == num_iterations - 1:
            final_ste_boundaries = ste_boundaries.detach().cpu()

        # Encourage the count of boundaries using the straight-through estimates for gradient flow
        # if L > 0:
        #     ste_counts = ste_boundaries.sum(dim=1).to(dtype=torch.float32)
        #     total_positions = torch.full_like(
        #         ste_counts, fill_value=float(L), dtype=torch.float32
        #     )
        #     binomial_loss = boundarypredictor_binomial_loss(
        #         num_boundaries=ste_counts,
        #         total_positions=total_positions,
        #         prior=binomial_target_rate,
        #     )
        # else:
        #     binomial_loss = torch.tensor(0.0, device=learned_logits.device)

        # Forward pass with finite difference gradients
        debug_this_step = USE_FAST_VERSION and (
            iteration == num_iterations - 1)
        output = downsample_with_grad(
            ste_boundaries,
            hidden,
            gradient_scale=1.0,
            debug=debug_this_step,
            debug_threshold=GRAD_ZERO_THRESHOLD,
        )

        # MSE error is super fuvkef up
        # avoid boundaries influencing the same posiitions with gradients
        # Compute all grad scores with all positions for super math

        # Pad outputs to the same length for loss calculation
        S_out = output.size(0)
        S_target = target_output.size(0)

        if S_out != S_target:
            max_S = max(S_out, S_target)
            output_padded = torch.zeros(max_S, B, D, device=output.device)
            target_padded = torch.zeros(
                max_S, B, D, device=target_output.device)
            output_padded[:S_out] = output
            target_padded[:S_target] = target_output
        else:
            output_padded = output
            target_padded = target_output

        # MSE loss between outputs
        mse_loss = F.mse_loss(output_padded, target_padded)
        loss = mse_loss  # + 100 * binomial_loss

        loss.backward()

        if target_ascii is not None and learned_logits.grad is not None:
            grad_values = learned_logits.grad[0].detach().cpu().tolist()
            grad_marker_str = ''.join(
                '0' if abs(val) < GRAD_ZERO_THRESHOLD else (
                    '+' if val > 0 else '-')
                for val in grad_values
            )
        else:
            grad_marker_str = None

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [learned_logits], max_norm=5.0)

        optimizer.step()

        if (iteration + 1) % log_every == 0 or iteration == 0:
            with torch.no_grad():
                # Report metrics based on the sampled hard boundaries this step
                current_hard = hard_boundaries.detach()
                learned_indices = current_hard[0].nonzero().squeeze(-1)
                num_learned = len(learned_indices)

                # Position-wise overlap
                learned_list = learned_indices.tolist()
                overlap = sum(idx in target_idx_set for idx in learned_list)
                overlap_pct = (overlap / K) * 100 if K > 0 else 0

                # Average distance of learned boundaries to nearest target boundary
                if num_learned > 0:
                    min_distances = []
                    for learned_idx in learned_indices:
                        distances = torch.abs(
                            target_boundary_indices.float() - learned_idx.float())
                        min_dist = distances.min().item()
                        min_distances.append(min_dist)
                    avg_min_distance = sum(min_distances) / len(min_distances)
                else:
                    avg_min_distance = float('inf')

            postfix_parts = [
                f"iter={iteration+1}",
                f"loss={loss.item():.6f}",
                f"mse={mse_loss.item():.6f}",
                f"grad={float(grad_norm):.4f}",
                f"overlap={overlap}/{num_learned}",
                f"dist={avg_min_distance:.2f}",
            ]

            if target_ascii is not None:
                learned_ascii = ''.join('|' if idx in learned_list else ' '
                                        for idx in range(L))
                if grad_marker_str is not None:
                    postfix_parts.append(f"G:{grad_marker_str}")
                postfix_parts.append(f"L:{learned_ascii}")
                postfix_parts.append(f"T:{target_ascii}")

            progress_bar.set_postfix_str(
                ' '.join(postfix_parts), refresh=False)

    progress_bar.close()

    final_logits_vec = learned_logits.detach().cpu().view(-1)
    final_logits_str = ' '.join(f"{val:.4f}" for val in final_logits_vec)
    print("\nFinal logits (per position):")
    print(final_logits_str)

    if learned_logits.grad is not None:
        final_grad_vec = learned_logits.grad.detach().cpu().view(-1)
        final_grad_str = ' '.join(f"{val:.4f}" for val in final_grad_vec)
        print("\nFinal logits gradients:")
        print(final_grad_str)
    else:
        print("\nFinal logits gradients: None (gradient not captured)")

    print("\n" + "=" * 80)
    print("Final Results")
    print("=" * 80)

    with torch.no_grad():
        final_boundaries = torch.zeros_like(learned_logits)
        if K > 1:
            _, final_topk = torch.topk(
                learned_logits[:, :-1], k=K - 1, dim=1, sorted=False)
            final_boundaries.scatter_(1, final_topk, 1.0)
        final_boundaries[:, -1] = 1.0

        final_indices = final_boundaries[0].nonzero().squeeze(-1).sort()[0]
        final_list = final_indices.tolist()

        # Calculate metrics
        overlap = sum(idx in target_idx_set for idx in final_list)
        overlap_pct = (overlap / K) * 100

        # Average distance
        min_distances = []
        for learned_idx in final_indices:
            distances = torch.abs(
                target_boundary_indices.float() - learned_idx.float())
            min_dist = distances.min().item()
            min_distances.append(min_dist)
        avg_min_distance = sum(min_distances) / len(min_distances)

        print(f"Overlap: {overlap}/{K} boundaries ({overlap_pct:.1f}%)")
        print(
            f"Average distance to nearest target: {avg_min_distance:.2f} timesteps")

        print(
            f"\nTarget boundary indices (first 20): {target_boundary_indices[:20].tolist()}")
        print(
            f"Learned boundary indices (first 20): {final_indices[:20].tolist()}")

        final_grad_markers = None
        if target_ascii is not None:
            final_ascii = ''.join(
                '|' if idx in final_list else ' ' for idx in range(L))
            if learned_logits.grad is not None:
                final_grad_markers = ''.join(
                    '0' if abs(val) < GRAD_ZERO_THRESHOLD else (
                        '+' if val > 0 else '-')
                    for val in learned_logits.grad[0].detach().cpu().tolist()
                )

            if final_grad_markers is not None:
                print(f"Grad:    {final_grad_markers}")
            print(f"Learned: {final_ascii}")
            print(f"Target:  {target_ascii}")

        if overlap_pct > 80:
            print("\n✓ Successfully learned boundaries with >80% overlap!")
        elif overlap_pct > 50:
            print(f"\n⚠ Partial success - {overlap_pct:.1f}% overlap")
        else:
            print(
                f"\n✗ Low overlap - {overlap_pct:.1f}%. May need more iterations or tuning.")

        if final_ste_boundaries is not None:
            final_boundary_indices = final_ste_boundaries[0].nonzero(as_tuple=True)[
                0].tolist()
            print(
                f"\nFinal hard boundaries (indices): {final_boundary_indices}")


if __name__ == "__main__":
    test_finite_difference_downsample(
        local_similarity_window=1,
        compression_rate=1.0 / 12.0,
        sequence_length=100,
    )

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
