import torch
from utils import downsample
from fast_downsample import FastFiniteDifferenceDownsample

def test_forward_pass_behavior():
    """Tests how the forward pass of `utils.downsample` interprets boundaries."""
    print("\n" + "="*80)
    print("1. Testing Forward Pass: How does `utils.downsample` work?")
    print("="*80)

    hidden = torch.tensor([10., 20., 30., 40.]).view(4, 1, 1)
    boundaries_end_of_segment = torch.tensor([0., 1., 0., 1.]).view(1, 4)
    output_B = downsample(boundaries_end_of_segment, hidden)

    print("\nTest Case: Assuming boundaries END a segment (e.g., [0, 1, 0, 1])")
    print(f"Input hidden states: {[h.item() for h in hidden]}")
    print(f"Output means: {[o.item() for o in output_B]}")

    expected = torch.tensor([15., 35.])
    if torch.allclose(output_B.squeeze(), expected):
        print("✓ RESULT: Correct! `utils.downsample` treats a 1 as the END of a segment.")
    else:
        print("✗ RESULT: Incorrect. The output does not match the expected means.")

def run_magnitude_test(num_tens):
    """Runs a test case to observe gradient magnitude with varying segment lengths."""
    print("\n" + "-"*80)
    print(f"Running magnitude test with a segment of {num_tens} 10s and one 100")
    print("-"*80)

    # Setup: A segment of `num_tens` 10s, one 100, and a second segment of 50s.
    hidden_list = [10.] * num_tens + [100.] + [50.] * 5
    L = len(hidden_list)
    hidden = torch.tensor(hidden_list).view(L, 1, 1)

    # The boundary is placed to incorrectly include the 100 in the first segment.
    boundary_pos = num_tens
    boundaries = torch.zeros(1, L)
    boundaries[0, boundary_pos] = 1
    boundaries[0, L - 1] = 1

    # Ideal output if the boundary were at `num_tens - 1`.
    target_output = torch.tensor([10., 50.]).view(2, 1, 1)

    # --- Forward and Backward Pass ---
    output = downsample(boundaries, hidden)
    grad_output = output - target_output

    class DummyCtx:
        def __init__(self):
            self.saved_tensors = (boundaries, hidden)
            self.gradient_scale = 1.0
    
    ctx = DummyCtx()
    grad_boundaries, _, _ = FastFiniteDifferenceDownsample.backward(ctx, grad_output)

    print(f"Segment 1 mean: {output.squeeze()[0]:.2f} (Target: 10)")
    print("Analysis: The BEST move is to shift the boundary at t={boundary_pos} LEFT to t={boundary_pos-1}.")
    print("Expected Gradient: Should PUSH AWAY from t={boundary_pos} and PULL TOWARDS t={boundary_pos-1}.")
    
    # Format the full gradient for printing
    grad_list = [f"{g:.1f}" for g in grad_boundaries.squeeze().tolist()]
    print(f"Computed Boundary Gradient: [{', '.join(grad_list)}]")

    grad_at_boundary = grad_boundaries[0, boundary_pos]
    grad_at_prev_step = grad_boundaries[0, boundary_pos - 1]

    if grad_at_boundary > 0 and grad_at_prev_step < 0:
        print("\n✓ RESULT: Correct direction! The gradient pushes away from the wrong spot and pulls towards the right one.")
    else:
        print("\n✗ RESULT: Incorrect gradient direction.")


if __name__ == "__main__":
    test_forward_pass_behavior()
    
    print("\n" + "="*80)
    print("2. Testing Gradient Magnitude vs. Segment Length")
    print("="*80)
    run_magnitude_test(num_tens=2)
    run_magnitude_test(num_tens=5)
    run_magnitude_test(num_tens=10)
    print("\n" + "="*80)
