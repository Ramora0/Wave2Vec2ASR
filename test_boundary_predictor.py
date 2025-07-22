#!/usr/bin/env python3
"""
Test script for BoundaryPredictor2 to verify it's working correctly.
"""

import torch
import torch.nn as nn
from BoundaryPredictor2 import BoundaryPredictor2


def test_boundary_predictor():
    print("Testing BoundaryPredictor2...")
    print("=" * 50)

    # Test parameters
    batch_size = 2
    seq_len = 8
    input_dim = 64
    hidden_dim = 32
    prior = 0.3
    temp = 1.0
    threshold = 0.5
    num_heads = 1

    print(f"Test configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  input_dim: {input_dim}")
    print(f"  prior: {prior}")
    print(f"  temp: {temp}")
    print(f"  threshold: {threshold}")
    print()

    # Create model
    model = BoundaryPredictor2(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        prior=prior,
        temp=temp,
        threshold=threshold,
        num_heads=num_heads
    )

    print(f"Model parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    # Create random input
    torch.manual_seed(42)  # For reproducible results
    hidden = torch.randn(batch_size, seq_len, input_dim)
    print(f"Input tensor shape: {hidden.shape}")
    print(f"Input tensor range: [{hidden.min():.4f}, {hidden.max():.4f}]")
    print()

    # Forward pass
    print("Running forward pass...")
    print("-" * 30)

    with torch.no_grad():
        pooled, loss = model(hidden)

    print("-" * 30)
    print("Forward pass complete!")
    print()

    # Check outputs
    print(f"Output shapes:")
    print(f"  pooled: {pooled.shape}")
    print(f"  loss: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
    print()

    print(f"Output ranges:")
    print(f"  pooled: [{pooled.min():.4f}, {pooled.max():.4f}]")
    print(f"  loss: {loss.item():.6f}")
    print()

    # Test gradient flow
    print("Testing gradient flow...")
    model.train()
    hidden.requires_grad_(True)

    pooled, loss = model(hidden)
    loss.backward()

    grad_norms = []
    param_names = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            param_names.append(name)
            print(f"  {name}: grad_norm = {grad_norm:.6f}")

    print(f"  input grad_norm: {hidden.grad.norm().item():.6f}")
    print()

    # Test multiple sequence lengths
    print("Testing different sequence lengths...")
    for test_seq_len in [4, 16, 32]:
        test_hidden = torch.randn(1, test_seq_len, input_dim)
        try:
            with torch.no_grad():
                test_pooled, test_loss = model(test_hidden)
            print(
                f"  seq_len={test_seq_len}: ✓ (output shape: {test_pooled.shape})")
        except Exception as e:
            print(f"  seq_len={test_seq_len}: ✗ Error: {e}")

    print()
    print("Test completed successfully! ✓")


if __name__ == "__main__":
    test_boundary_predictor()
