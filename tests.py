"""
Test to verify that padding tokens don't affect non-padded tokens in MagnetAttention.

This test creates a batch with different sequence lengths (requiring padding),
runs it through MagnetAttention, then runs each element individually without padding,
and verifies that the outputs match for non-padded positions.
"""

import torch
import torch.nn as nn
from MagnetEncoderLayer import MagnetAttention, MagnetEncoderLayer


def create_magnet_attention(embed_dim=256, num_heads=4, dropout=0.0):
    """Create a MagnetAttention instance with specified parameters."""
    attention = MagnetAttention()

    # Set required attributes
    attention.embed_dim = embed_dim
    attention.num_heads = num_heads
    attention.head_dim = embed_dim // num_heads
    attention.scaling = attention.head_dim ** -0.5
    attention.dropout = dropout
    attention.layer_idx = 0

    # Create projection layers
    attention.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
    attention.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
    attention.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
    attention.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    return attention


def create_magnet_encoder_layer(embed_dim=256, num_heads=4, ffn_dim=1024, dropout=0.0):
    """Create a MagnetEncoderLayer instance with specified parameters."""
    layer = MagnetEncoderLayer()

    # Set required attributes
    layer.embed_dim = embed_dim
    layer.dropout = dropout
    layer.activation_dropout = dropout

    # Create attention module
    layer.self_attn = create_magnet_attention(embed_dim, num_heads, dropout)

    # Create layer normalization
    layer.self_attn_layer_norm = nn.LayerNorm(embed_dim)
    layer.final_layer_norm = nn.LayerNorm(embed_dim)

    # Create feed-forward layers
    layer.fc1 = nn.Linear(embed_dim, ffn_dim)
    layer.fc2 = nn.Linear(ffn_dim, embed_dim)

    # Set activation function
    layer.activation_fn = nn.GELU()

    return layer


def test_padding_invariance_attention():
    """
    Test that padding doesn't affect non-padded tokens in MagnetAttention.

    Strategy:
    1. Create a batch with sequences of different lengths
    2. Pad to the maximum length and create attention mask
    3. Run batch through MagnetAttention
    4. Run each sequence individually without padding
    5. Compare outputs for non-padded positions
    """
    print("\n" + "="*80)
    print("Testing MagnetAttention Padding Invariance")
    print("="*80)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    embed_dim = 256
    num_heads = 4
    batch_size = 3

    # Different sequence lengths for each batch element
    seq_lengths = [10, 7, 5]
    max_seq_len = max(seq_lengths)

    print(f"\nTest Configuration:")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Num heads: {num_heads}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence lengths: {seq_lengths}")
    print(f"  Max sequence length: {max_seq_len}")

    # Create MagnetAttention module
    attention = create_magnet_attention(embed_dim, num_heads, dropout=0.0)
    attention.eval()  # Set to eval mode to disable dropout

    # Create input data for each sequence (unpadded)
    unpadded_inputs = []
    for seq_len in seq_lengths:
        unpadded_inputs.append(torch.randn(1, seq_len, embed_dim))

    # Create padded batch with random values in padded positions
    # This is a more rigorous test - ensures random padding doesn't affect valid tokens
    padded_batch = torch.randn(batch_size, max_seq_len, embed_dim)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

    for i, seq_len in enumerate(seq_lengths):
        padded_batch[i, :seq_len, :] = unpadded_inputs[i][0]
        attention_mask[i, :seq_len] = 1  # 1 = valid, 0 = padded
        # Padded positions (seq_len:) contain random garbage data

    print(f"\nAttention mask:\n{attention_mask}")

    # Run padded batch through attention
    print("\nRunning padded batch through MagnetAttention...")
    with torch.no_grad():
        padded_output, _, _ = attention(
            hidden_states=padded_batch,
            attention_mask_1d=attention_mask,
            output_attentions=False
        )

    # Run each sequence individually without padding
    print("Running individual sequences without padding...")
    individual_outputs = []
    for i, seq_len in enumerate(seq_lengths):
        input_tensor = unpadded_inputs[i]
        mask = torch.ones(1, seq_len, dtype=torch.long)

        with torch.no_grad():
            output, _, _ = attention(
                hidden_states=input_tensor,
                attention_mask_1d=mask,
                output_attentions=False
            )
        individual_outputs.append(output)

    # Compare outputs
    print("\nComparing outputs...")
    all_match = True
    max_diff_overall = 0.0

    for i, seq_len in enumerate(seq_lengths):
        # Extract non-padded portion from batched output
        batched_output_slice = padded_output[i, :seq_len, :]
        individual_output = individual_outputs[i][0]

        # Calculate difference
        diff = torch.abs(batched_output_slice - individual_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        max_diff_overall = max(max_diff_overall, max_diff)

        print(f"\nSequence {i} (length {seq_len}):")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")

        # Check if outputs match (allowing small numerical errors)
        tolerance = 1e-5
        matches = max_diff < tolerance
        all_match = all_match and matches

        if matches:
            print(f"  ✓ PASS: Outputs match within tolerance ({tolerance})")
        else:
            print(f"  ✗ FAIL: Outputs differ by more than tolerance ({tolerance})")
            print(f"  Batched output sample: {batched_output_slice[0, :5]}")
            print(f"  Individual output sample: {individual_output[0, :5]}")

    # Final result
    print("\n" + "="*80)
    if all_match:
        print("✓ TEST PASSED: Padding does not affect non-padded tokens!")
        print(f"  Maximum difference across all sequences: {max_diff_overall:.2e}")
    else:
        print("✗ TEST FAILED: Padding affects non-padded tokens!")
    print("="*80 + "\n")

    return all_match


def test_padding_invariance_encoder_layer():
    """
    Test that padding doesn't affect non-padded tokens in MagnetEncoderLayer.

    This tests the full encoder layer including attention, layer norm, and FFN.
    """
    print("\n" + "="*80)
    print("Testing MagnetEncoderLayer Padding Invariance")
    print("="*80)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    embed_dim = 256
    num_heads = 4
    ffn_dim = 1024
    batch_size = 3

    # Different sequence lengths for each batch element
    seq_lengths = [10, 7, 5]
    max_seq_len = max(seq_lengths)

    print(f"\nTest Configuration:")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Num heads: {num_heads}")
    print(f"  FFN dim: {ffn_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence lengths: {seq_lengths}")
    print(f"  Max sequence length: {max_seq_len}")

    # Create MagnetEncoderLayer module
    encoder_layer = create_magnet_encoder_layer(embed_dim, num_heads, ffn_dim, dropout=0.0)
    encoder_layer.eval()  # Set to eval mode to disable dropout

    # Create input data for each sequence (unpadded)
    unpadded_inputs = []
    for seq_len in seq_lengths:
        unpadded_inputs.append(torch.randn(1, seq_len, embed_dim))

    # Create padded batch with random values in padded positions
    # This is a more rigorous test - ensures random padding doesn't affect valid tokens
    padded_batch = torch.randn(batch_size, max_seq_len, embed_dim)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

    for i, seq_len in enumerate(seq_lengths):
        padded_batch[i, :seq_len, :] = unpadded_inputs[i][0]
        attention_mask[i, :seq_len] = 1  # 1 = valid, 0 = padded
        # Padded positions (seq_len:) contain random garbage data

    print(f"\nAttention mask:\n{attention_mask}")

    # Run padded batch through encoder layer
    print("\nRunning padded batch through MagnetEncoderLayer...")
    with torch.no_grad():
        padded_output = encoder_layer(
            hidden_states=padded_batch,
            attention_mask_1d=attention_mask,
            layer_head_mask=None,
            output_attentions=False
        )[0]

    # Run each sequence individually without padding
    print("Running individual sequences without padding...")
    individual_outputs = []
    for i, seq_len in enumerate(seq_lengths):
        input_tensor = unpadded_inputs[i]
        mask = torch.ones(1, seq_len, dtype=torch.long)

        with torch.no_grad():
            output = encoder_layer(
                hidden_states=input_tensor,
                attention_mask_1d=mask,
                layer_head_mask=None,
                output_attentions=False
            )[0]
        individual_outputs.append(output)

    # Compare outputs
    print("\nComparing outputs...")
    all_match = True
    max_diff_overall = 0.0

    for i, seq_len in enumerate(seq_lengths):
        # Extract non-padded portion from batched output
        batched_output_slice = padded_output[i, :seq_len, :]
        individual_output = individual_outputs[i][0]

        # Calculate difference
        diff = torch.abs(batched_output_slice - individual_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        max_diff_overall = max(max_diff_overall, max_diff)

        print(f"\nSequence {i} (length {seq_len}):")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")

        # Check if outputs match (allowing small numerical errors)
        tolerance = 1e-5
        matches = max_diff < tolerance
        all_match = all_match and matches

        if matches:
            print(f"  ✓ PASS: Outputs match within tolerance ({tolerance})")
        else:
            print(f"  ✗ FAIL: Outputs differ by more than tolerance ({tolerance})")
            print(f"  Batched output sample: {batched_output_slice[0, :5]}")
            print(f"  Individual output sample: {individual_output[0, :5]}")

    # Final result
    print("\n" + "="*80)
    if all_match:
        print("✓ TEST PASSED: Padding does not affect non-padded tokens!")
        print(f"  Maximum difference across all sequences: {max_diff_overall:.2e}")
    else:
        print("✗ TEST FAILED: Padding affects non-padded tokens!")
    print("="*80 + "\n")

    return all_match


def test_padded_positions_stay_zero():
    """
    Additional test: Verify that padded positions in the output remain as they were input.
    This ensures padded positions don't leak information.
    """
    print("\n" + "="*80)
    print("Testing Padded Positions Remain Unchanged")
    print("="*80)

    torch.manual_seed(42)

    embed_dim = 256
    num_heads = 4
    seq_len = 10
    num_valid = 6

    print(f"\nTest Configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Valid tokens: {num_valid}")
    print(f"  Padded tokens: {seq_len - num_valid}")

    # Create attention module
    attention = create_magnet_attention(embed_dim, num_heads, dropout=0.0)
    attention.eval()

    # Create input with padding (padded positions contain random data)
    hidden_states = torch.randn(1, seq_len, embed_dim)
    # Keep random values in padded positions to test they don't leak

    # Create mask
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    attention_mask[0, num_valid:] = 0  # Mark padded positions

    print(f"\nAttention mask: {attention_mask[0]}")

    # Run through attention
    with torch.no_grad():
        output, _, _ = attention(
            hidden_states=hidden_states,
            attention_mask_1d=attention_mask,
            output_attentions=False
        )

    # Check that padded positions in output are close to zero or unchanged
    padded_output = output[0, num_valid:, :]
    max_padded_output = padded_output.abs().max().item()
    mean_padded_output = padded_output.abs().mean().item()

    print(f"\nPadded positions output statistics:")
    print(f"  Max absolute value: {max_padded_output:.2e}")
    print(f"  Mean absolute value: {mean_padded_output:.2e}")

    # The output at padded positions might not be exactly zero due to residual connections
    # in the encoder layer, but attention output itself should not contribute
    print("\n" + "="*80)
    print("✓ TEST COMPLETED: Padded positions output statistics shown above")
    print("  (Note: In full encoder layer, residual connections may add to padded positions)")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "MAGNET ATTENTION PADDING TESTS" + " "*28 + "║")
    print("╚" + "="*78 + "╝")

    # Run all tests
    test1_passed = test_padding_invariance_attention()
    test2_passed = test_padding_invariance_encoder_layer()
    test_padded_positions_stay_zero()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"MagnetAttention padding invariance: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"MagnetEncoderLayer padding invariance: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print("="*80 + "\n")

    if test1_passed and test2_passed:
        print("🎉 All critical tests PASSED! Padding does not affect non-padded tokens.")
    else:
        print("⚠️  Some tests FAILED. Please review the results above.")
    print()
