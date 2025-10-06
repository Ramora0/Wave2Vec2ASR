import torch
from utils import downsample


def test_downsample_with_start_end_boundaries():
    """
    Test downsample with boundaries that have 1 at start and end, with trailing 0s.
    This mimics the real usage pattern from BoundaryPredictor1.
    """
    print("\n=== Test: Boundaries with 1 at start and end, trailing zeros ===\n")

    # Create a simple test case
    batch_size = 2
    seq_len = 8
    model_dim = 4

    # Boundaries: 1 at positions that mark segment boundaries
    # Example: [1, 0, 0, 1, 0, 1, 0, 0] means segments at positions 0, 3, 5
    boundaries = torch.zeros(batch_size, seq_len, dtype=torch.float32)

    # Batch 0: boundaries at positions 0, 3, 5
    boundaries[0, 0] = 1.0
    boundaries[0, 3] = 1.0
    boundaries[0, 5] = 1.0

    # Batch 1: boundaries at positions 0, 2, 6
    boundaries[1, 0] = 1.0
    boundaries[1, 2] = 1.0
    boundaries[1, 6] = 1.0

    print("Boundaries (B x L):")
    print(boundaries)
    print()

    # Create simple hidden states for easy verification
    hidden = torch.arange(batch_size * seq_len *
                          model_dim, dtype=torch.float32)
    hidden = hidden.reshape(batch_size, seq_len, model_dim)

    print("Hidden states (B x L x D):")
    print(hidden)
    print()

    # Transpose to L x B x D as expected by downsample
    hidden_transposed = hidden.transpose(0, 1)  # L x B x D

    # Call downsample
    pooled = downsample(boundaries, hidden_transposed)

    print("Pooled output (S x B x D):")
    print(pooled)
    print(f"Pooled shape: {pooled.shape}")
    print()

    # Verify shape
    max_segments = int(boundaries.sum(dim=1).max().item())
    print(f"Max segments across batch: {max_segments}")
    print(
        f"Expected output shape: ({max_segments} x {batch_size} x {model_dim})")
    print(f"Actual output shape: {tuple(pooled.shape)}")
    print()

    # Manual verification for batch 0
    # Boundaries at [0, 3, 5]
    # Segment 0: indices [0:1] -> hidden[0, 0:1].mean(0)
    # Segment 1: indices [1:4] -> hidden[0, 1:4].mean(0)
    # Segment 2: indices [4:6] -> hidden[0, 4:6].mean(0)

    print("Manual verification for batch 0:")
    print(f"Segment 0 (indices 0:1): mean = {hidden[0, 0:1].mean(0)}")
    print(f"Segment 1 (indices 1:4): mean = {hidden[0, 1:4].mean(0)}")
    print(f"Segment 2 (indices 4:6): mean = {hidden[0, 4:6].mean(0)}")
    print()

    print("Actual pooled values for batch 0:")
    for i in range(pooled.shape[0]):
        print(f"pooled[{i}, 0] = {pooled[i, 0]}")
    print()

    print("=== Test complete ===")


def test_edge_cases():
    """Test various edge cases"""
    print("\n=== Test: Edge Cases ===\n")

    batch_size = 1
    seq_len = 6
    model_dim = 3

    # Test 1: Only one boundary at position 0
    print("Test 1: Single boundary at start")
    boundaries = torch.zeros(batch_size, seq_len)
    boundaries[0, 0] = 1.0

    hidden = torch.randn(seq_len, batch_size, model_dim)

    pooled = downsample(boundaries, hidden)
    print(f"Boundaries: {boundaries[0].tolist()}")
    print(f"Pooled shape: {pooled.shape}")
    print(f"Pooled:\n{pooled}")
    print()

    # Test 2: Multiple boundaries
    print("Test 2: Multiple boundaries")
    boundaries = torch.zeros(batch_size, seq_len)
    boundaries[0, 0] = 1.0
    boundaries[0, 2] = 1.0
    boundaries[0, 4] = 1.0

    pooled = downsample(boundaries, hidden)
    print(f"Boundaries: {boundaries[0].tolist()}")
    print(f"Pooled shape: {pooled.shape}")
    print(f"Pooled:\n{pooled}")
    print()

    # Test 3: All zeros (no boundaries)
    print("Test 3: No boundaries (all zeros)")
    boundaries = torch.zeros(batch_size, seq_len)

    pooled = downsample(boundaries, hidden)
    print(f"Boundaries: {boundaries[0].tolist()}")
    print(f"Pooled shape: {pooled.shape}")
    print(f"Pooled:\n{pooled}")
    print()


def test_specific_boundary_pattern():
    """Test boundaries pattern 10101000 to understand exact pooling behavior"""
    print("\n=== Test: Boundaries [1, 0, 1, 0, 1, 0, 0, 0] ===\n")

    batch_size = 1
    seq_len = 8
    model_dim = 1

    # Boundaries: 10101000
    boundaries = torch.tensor([[1., 0., 1., 0., 1., 0., 0., 0.]])

    # Simple hidden states - just use the index as the value for easy tracking
    hidden = torch.arange(seq_len, dtype=torch.float32).reshape(1, seq_len, 1)

    print("Boundaries:", boundaries[0].tolist())
    print("Hidden (values = indices):", hidden[0].squeeze().tolist())
    print()

    # Transpose to L x B x D
    hidden_transposed = hidden.transpose(0, 1)

    # Call downsample
    pooled = downsample(boundaries, hidden_transposed)

    print(f"Number of outputs: {pooled.shape[0]}")
    print(f"Number of boundaries (1s): {boundaries.sum().item()}")
    print()

    print("Pooled output (S x B x D):")
    for i in range(pooled.shape[0]):
        print(f"  Output[{i}] = {pooled[i, 0, 0].item()}")
    print()

    print("Understanding the segments:")
    print("  - A boundary of 1 at position i means: segment ENDS at position i (inclusive)")
    print()

    num_boundaries = int(boundaries.sum().item())
    print(f"With {num_boundaries} boundaries, we get {num_boundaries} segments (outputs)")
    print()

    print("Manual segment calculation:")
    print(
        "  Segment 0 (boundary at pos 0): indices [0:1]   -> mean([0]) = 0.0")
    print(
        "  Segment 1 (boundary at pos 2): indices [1:3]   -> mean([1, 2]) = 1.5")
    print(
        "  Segment 2 (boundary at pos 4): indices [3:5]   -> mean([3, 4]) = 3.5")
    print()
    print("  Note: Positions 5, 6, 7 have no boundary, so they're not included in any segment!")
    print()


if __name__ == "__main__":
    test_specific_boundary_pattern()
    # test_downsample_with_start_end_boundaries()
    # test_edge_cases()
