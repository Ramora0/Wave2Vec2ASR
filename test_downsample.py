"""
Test script to verify downsample behavior with all-1 boundaries.
When all boundaries are 1, downsample should be a perfect no-op.
"""

import torch
from utils import downsample


def test_downsample_all_ones():
    """Test that downsample with all 1s preserves the original vectors."""

    # Test configuration
    batch_size = 2
    seq_len = 10
    hidden_dim = 768

    print("=" * 60)
    print("Testing downsample with all-1 boundaries")
    print("=" * 60)

    # Create test data
    torch.manual_seed(42)
    hidden = torch.randn(batch_size, seq_len, hidden_dim)  # B x L x D
    boundaries = torch.ones(batch_size, seq_len)  # All 1s
    attention_mask = torch.ones(batch_size, seq_len)  # All valid

    print(f"\nInput shapes:")
    print(f"  hidden: {hidden.shape}")
    print(f"  boundaries: {boundaries.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  boundaries sum per sample: {boundaries.sum(dim=1).tolist()}")

    # Run downsample (needs L x B x D format)
    hidden_transposed = hidden.transpose(0, 1)  # L x B x D
    pooled_transposed = downsample(
        boundaries,
        hidden_transposed,
        attention_mask=attention_mask
    )
    pooled = pooled_transposed.transpose(0, 1)  # B x L x D

    print(f"\nOutput shape: {pooled.shape}")

    # Check if shapes match
    if hidden.shape != pooled.shape:
        print(f"\n❌ SHAPE MISMATCH: {hidden.shape} -> {pooled.shape}")
        print("With all boundaries = 1, shapes should be identical!")
        return False

    # Check if values match
    max_diff = (hidden - pooled).abs().max().item()
    mean_diff = (hidden - pooled).abs().mean().item()

    print(f"\nDifference statistics:")
    print(f"  Max diff: {max_diff}")
    print(f"  Mean diff: {mean_diff}")

    if max_diff > 1e-5:
        print(f"\n❌ VALUE MISMATCH: max diff = {max_diff}")
        print("With all boundaries = 1, values should be identical!")

        # Show some example differences
        print("\nExample differences (first 5 positions):")
        for i in range(min(5, seq_len)):
            diff = (hidden[0, i, :] - pooled[0, i, :]).abs().max().item()
            print(f"  Position {i}: max diff = {diff}")

        return False

    print(f"\n✅ PASS: Downsample correctly preserves vectors when all boundaries = 1")
    return True


def test_downsample_with_padding():
    """Test downsample with padded sequences (mixed attention masks)."""

    print("\n" + "=" * 60)
    print("Testing downsample with padding")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    hidden_dim = 1

    # Create test data with variable-length sequences
    torch.manual_seed(42)
    hidden = torch.randn(batch_size, seq_len, hidden_dim)
    print(hidden)
    boundaries = torch.ones(batch_size, seq_len)  # All 1s initially

    # Sample 1: length 7, Sample 2: length 5
    attention_mask = torch.zeros(batch_size, seq_len)
    attention_mask[0, :7] = 1
    attention_mask[1, :5] = 1

    # Zero out boundaries in padded regions
    boundaries = boundaries * attention_mask

    print(f"\nSequence lengths: [7, 5]")
    print(f"Boundaries per sample: {boundaries.sum(dim=1).tolist()}")

    # Run downsample
    hidden_transposed = hidden.transpose(0, 1)
    pooled_transposed = downsample(
        boundaries,
        hidden_transposed,
        attention_mask=attention_mask
    )
    pooled = pooled_transposed.transpose(0, 1)
    print(pooled)

    print(f"\nOutput shape: {pooled.shape}")
    print(f"Expected shape: [{batch_size}, 7, {hidden_dim}] (max length)")

    # For sample 1 (length 7), first 7 positions should match
    trimmed_hidden_0 = hidden[0, :7, :]
    pooled_0 = pooled[0, :7, :]
    max_diff_0 = (trimmed_hidden_0 - pooled_0).abs().max().item()

    # For sample 2 (length 5), first 5 positions should match
    trimmed_hidden_1 = hidden[1, :5, :]
    pooled_1 = pooled[1, :5, :]
    max_diff_1 = (trimmed_hidden_1 - pooled_1).abs().max().item()

    print(f"\nDifferences for valid positions:")
    print(f"  Sample 1 (7 tokens): max diff = {max_diff_0}")
    print(f"  Sample 2 (5 tokens): max diff = {max_diff_1}")

    if max_diff_0 > 1e-5 or max_diff_1 > 1e-5:
        print(f"\n❌ VALUE MISMATCH in padded case")
        return False

    print(f"\n✅ PASS: Downsample handles padding correctly")
    return True


def test_downsample_fp16():
    """Test downsample with FP16 (half precision)."""

    print("\n" + "=" * 60)
    print("Testing downsample with FP16")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    hidden_dim = 768

    # Create FP16 test data
    torch.manual_seed(42)
    hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16)
    boundaries = torch.ones(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)

    print(f"\nInput dtype: {hidden.dtype}")

    # Run downsample
    hidden_transposed = hidden.transpose(0, 1)
    pooled_transposed = downsample(
        boundaries,
        hidden_transposed,
        attention_mask=attention_mask
    )
    pooled = pooled_transposed.transpose(0, 1)

    print(f"Output dtype: {pooled.dtype}")

    if pooled.dtype != torch.float16:
        print(f"\n❌ DTYPE MISMATCH: expected float16, got {pooled.dtype}")
        return False

    # Check values (use float32 for comparison)
    max_diff = (hidden.float() - pooled.float()).abs().max().item()
    print(f"\nMax diff (in FP32): {max_diff}")

    if max_diff > 1e-3:  # More lenient for FP16
        print(f"\n❌ VALUE MISMATCH in FP16: max diff = {max_diff}")
        return False

    print(f"\n✅ PASS: Downsample preserves FP16 dtype and values")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DOWNSAMPLE FUNCTION TESTS")
    print("=" * 60 + "\n")

    results = []

    results.append(("All-1 boundaries", test_downsample_all_ones()))
    results.append(("Padding handling", test_downsample_with_padding()))
    results.append(("FP16 support", test_downsample_fp16()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed - downsample function has issues!")
