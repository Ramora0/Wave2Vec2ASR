import torch
import timeit
import random

from finite_difference_downsample import downsample_with_finite_diff_grad as slow_downsample
from fast_downsample import downsample_with_fast_finite_diff_grad as fast_downsample
from utils import downsample as util_downsample


def run_speed_test(seq_len=1500, embedding_dim=768, num_boundaries=125, batch_size=1):
    """Runs a speed test comparing the slow and fast implementations."""

    # Generate random data
    hidden_states = torch.randn(
        seq_len, batch_size, embedding_dim, requires_grad=True)

    # Generate random boundaries
    boundary_indices = torch.tensor(sorted(random.sample(
        range(seq_len - 1), num_boundaries - 1)) + [seq_len - 1])
    boundaries = torch.zeros(batch_size, seq_len)
    boundaries[0, boundary_indices] = 1

    print("=" * 80)
    print("Speed Comparison")
    print(
        f"(SeqLen={seq_len}, Dim={embedding_dim}, Boundaries={num_boundaries}, Batch={batch_size})")
    print("=" * 80)

    # --- Slow (Reference) Finite Difference Downsample ---
    hidden_slow = hidden_states.clone()
    boundaries_slow = boundaries.clone().detach()
    boundaries_slow.requires_grad = True

    start_time = timeit.default_timer()
    output_slow = slow_downsample(boundaries_slow, hidden_slow)
    forward_time_slow = timeit.default_timer() - start_time

    grad_output_slow = torch.randn_like(output_slow)
    start_time = timeit.default_timer()
    output_slow.backward(grad_output_slow)
    backward_time_slow = timeit.default_timer() - start_time

    # --- Fast (Analytical) Finite Difference Downsample ---
    hidden_fast = hidden_states.clone()
    boundaries_fast = boundaries.clone().detach()
    boundaries_fast.requires_grad = True

    start_time = timeit.default_timer()
    output_fast = fast_downsample(boundaries_fast, hidden_fast)
    forward_time_fast = timeit.default_timer() - start_time

    grad_output_fast = torch.randn_like(output_fast)
    start_time = timeit.default_timer()
    output_fast.backward(grad_output_fast)
    backward_time_fast = timeit.default_timer() - start_time

    # --- Baseline Util Downsample ---
    hidden_util = hidden_states.clone()
    boundaries_util = boundaries.clone().detach()
    boundaries_util.requires_grad = True  # Enable grad for backward pass

    start_time = timeit.default_timer()
    output_util = util_downsample(boundaries_util, hidden_util)
    forward_time_util = timeit.default_timer() - start_time

    grad_output_util = torch.randn_like(output_util)
    start_time = timeit.default_timer()
    output_util.backward(grad_output_util)
    backward_time_util = timeit.default_timer() - start_time

    # Report results
    print(f"\nSlow Implementation (Reference):")
    print(f"  Forward pass:  {forward_time_slow:.6f} seconds")
    print(f"  Backward pass: {backward_time_slow:.6f} seconds")
    print(
        f"  Total:         {forward_time_slow + backward_time_slow:.6f} seconds")

    print(f"\nFast Implementation (Analytical):")
    print(f"  Forward pass:  {forward_time_fast:.6f} seconds")
    print(f"  Backward pass: {backward_time_fast:.6f} seconds")
    print(
        f"  Total:         {forward_time_fast + backward_time_fast:.6f} seconds")

    print(f"\nUtil Downsample (Baseline with Autograd):")
    print(f"  Forward pass:  {forward_time_util:.6f} seconds")
    print(f"  Backward pass: {backward_time_util:.6f} seconds")
    print(
        f"  Total:         {forward_time_util + backward_time_util:.6f} seconds")
    print("  (Note: Backward pass uses standard autograd, not finite difference)")

    print("\n" + "=" * 80)
    print("Speedup Analysis (vs. Slow Reference)")
    print("=" * 80)

    # Avoid division by zero if a time is zero
    if forward_time_fast > 0:
        speedup_fwd = forward_time_slow / forward_time_fast
        print(f"  Forward Speedup: {speedup_fwd:.2f}x")

    if backward_time_fast > 0:
        speedup_bwd = backward_time_slow / backward_time_fast
        print(f"  Backward Speedup: {speedup_bwd:.2f}x")

    total_slow = forward_time_slow + backward_time_slow
    total_fast = forward_time_fast + backward_time_fast
    if total_fast > 0:
        speedup_total = total_slow / total_fast
        print(f"  Total Speedup:    {speedup_total:.2f}x")


if __name__ == "__main__":
    run_speed_test()
