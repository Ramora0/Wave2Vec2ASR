import torch
import unittest
import random
from finite_difference_downsample import downsample_with_finite_diff_grad, compute_pooled_state_diff_for_left_shift as slow_left, compute_pooled_state_diff_for_right_shift as slow_right
from fast_downsample import downsample_with_fast_finite_diff_grad, simulate_left_shift as fast_left, simulate_right_shift as fast_right
from utils import downsample


class TestImplementationComparison(unittest.TestCase):

    def setUp(self):
        """Set up common tensors for all tests."""
        self.B, self.L, self.D = 2, 20, 8
        self.hidden = torch.randn(self.L, self.B, self.D)

        boundaries = torch.zeros(self.B, self.L, dtype=torch.float32)
        boundaries[0, 0] = 1
        boundaries[0, 5] = 1
        boundaries[0, 12] = 1
        boundaries[1, 0] = 1
        boundaries[1, 8] = 1
        boundaries[1, 15] = 1
        self.boundaries = boundaries

    def test_boundary_shift_simulation(self):
        """Test that the boundary shift simulations are identical."""
        torch.manual_seed(0)
        random.seed(0)
        for k in range(100):
            B, L, D = random.randint(1, 4), random.randint(
                10, 50), random.randint(2, 16)
            hidden = torch.randn(L, B, D)
            boundaries = torch.zeros(B, L, dtype=torch.float32)

            for b_idx in range(B):
                boundaries[b_idx, 0] = 1
                num_boundaries = random.randint(1, L // 4)
                boundary_indices_rand = random.sample(
                    range(1, L), num_boundaries)
                for i_rand in boundary_indices_rand:
                    boundaries[b_idx, i_rand] = 1

            b = random.randint(0, B - 1)
            
            boundary_indices_for_b = boundaries[b].nonzero().squeeze().tolist()
            if isinstance(boundary_indices_for_b, int): # Handle single boundary case
                boundary_indices_for_b = [boundary_indices_for_b]
            if not boundary_indices_for_b:
                continue
            i = random.choice(boundary_indices_for_b)

            original_output = downsample(boundaries, hidden)

            # Find seg_idx for the chosen boundary `i`
            seg_idx = 0
            sorted_boundary_indices = sorted(boundary_indices_for_b)
            for boundary_pos in sorted_boundary_indices:
                seg_idx += 1
                if boundary_pos == i:
                    break
            
            # Test left shift
            slow_left_res = slow_left(boundaries, hidden, original_output, b, i, seg_idx)
            fast_left_res = fast_left(
                boundaries, hidden, b, i, original_output)
            if slow_left_res is None:
                self.assertIsNone(fast_left_res)
            else:
                self.assertIsNotNone(fast_left_res, f"FAILING CASE (left shift, iteration {k}): slow is not None but fast is")
                torch.testing.assert_close(slow_left_res[0], fast_left_res[0], msg=f"FAILING CASE (left shift, iter {k})")
                torch.testing.assert_close(slow_left_res[1], fast_left_res[1], msg=f"FAILING CASE (left shift, iter {k})")

            # Test right shift
            slow_right_res = slow_right(boundaries, hidden, original_output, b, i, seg_idx)
            fast_right_res = fast_right(
                boundaries, hidden, b, i, original_output)
            if slow_right_res is None:
                self.assertIsNone(fast_right_res)
            else:
                self.assertIsNotNone(fast_right_res, f"FAILING CASE (right shift, iteration {k}): slow is not None but fast is")
                torch.testing.assert_close(slow_right_res[0], fast_right_res[0], msg=f"FAILING CASE (right shift, iter {k})")
                torch.testing.assert_close(slow_right_res[1], fast_right_res[1], msg=f"FAILING CASE (right shift, iter {k})")

    def get_gradients(self, func, boundaries, hidden):
        boundaries = boundaries.clone().requires_grad_()
        hidden = hidden.clone()
        output = func(boundaries, hidden)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        return boundaries.grad.clone()

    def test_implementation_consistency(self):
        """Compares the reference gradient calculation with the optimized implementation."""
        reference_gradients = self.get_gradients(
            downsample_with_finite_diff_grad, self.boundaries, self.hidden)
        optimized_gradients = self.get_gradients(
            downsample_with_fast_finite_diff_grad, self.boundaries, self.hidden)

        print("\nReference Gradients:\n", reference_gradients)
        print("\nOptimized Gradients:\n", optimized_gradients)

        torch.testing.assert_close(reference_gradients, optimized_gradients)


if __name__ == '__main__':
    unittest.main()
