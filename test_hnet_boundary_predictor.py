import unittest
import torch
import torch.nn.functional as F
from HNetBoundaryPredictor import HNetBoundaryPredictor, WhisperBoundaryOutput, WhisperBoundaryState


class TestHNetBoundaryPredictor(unittest.TestCase):
    """Test suite for HNetBoundaryPredictor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.d_model = 512
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.batch_size = 2
        self.seq_len = 10

        # Initialize the boundary predictor
        self.predictor = HNetBoundaryPredictor(
            d_model=self.d_model,
            device=self.device,
            dtype=self.dtype
        )

        # Create sample input tensor
        self.hidden = torch.randn(
            self.batch_size, self.seq_len, self.d_model,
            device=self.device, dtype=self.dtype
        )

    def test_initialization(self):
        """Test proper initialization of the boundary predictor."""
        # Check model parameters
        self.assertEqual(self.predictor.d_model, self.d_model)

        # Check linear layers exist
        self.assertIsInstance(self.predictor.q_proj_layer, torch.nn.Linear)
        self.assertIsInstance(self.predictor.k_proj_layer, torch.nn.Linear)

        # Check that weights are initialized as identity matrices
        expected_identity = torch.eye(self.d_model)
        torch.testing.assert_close(
            self.predictor.q_proj_layer.weight.data,
            expected_identity,
            rtol=1e-5, atol=1e-8
        )
        torch.testing.assert_close(
            self.predictor.k_proj_layer.weight.data,
            expected_identity,
            rtol=1e-5, atol=1e-8
        )

        # Check no reinit flags
        self.assertTrue(
            hasattr(self.predictor.q_proj_layer.weight, '_no_reinit'))
        self.assertTrue(
            hasattr(self.predictor.k_proj_layer.weight, '_no_reinit'))

    def test_forward_output_structure(self):
        """Test that forward pass returns correct output structure."""
        output = self.predictor.forward(self.hidden)

        # Check output type
        self.assertIsInstance(output, WhisperBoundaryOutput)

        # Check output tensor shapes
        self.assertEqual(output.boundary_prob.shape,
                         (self.batch_size, self.seq_len, 2))
        self.assertEqual(output.boundary_mask.shape,
                         (self.batch_size, self.seq_len))
        self.assertEqual(output.selected_probs.shape,
                         (self.batch_size, self.seq_len, 1))

        # Compressed features should have variable length but correct batch size and d_model
        self.assertEqual(output.compressed_features.shape[0], self.batch_size)
        self.assertEqual(output.compressed_features.shape[2], self.d_model)

    def test_forward_output_values(self):
        """Test that forward pass produces valid output values."""
        output = self.predictor.forward(self.hidden)

        # Check boundary probabilities are valid probabilities
        self.assertTrue(torch.all(output.boundary_prob >= 0.0))
        self.assertTrue(torch.all(output.boundary_prob <= 1.0))

        # Check that probabilities sum to 1 across the last dimension
        prob_sums = torch.sum(output.boundary_prob, dim=-1)
        torch.testing.assert_close(
            prob_sums,
            torch.ones_like(prob_sums),
            rtol=1e-5, atol=1e-6
        )

        # Check boundary mask is boolean
        self.assertEqual(output.boundary_mask.dtype, torch.bool)

        # Check selected probabilities are valid
        self.assertTrue(torch.all(output.selected_probs >= 0.0))
        self.assertTrue(torch.all(output.selected_probs <= 1.0))

    def test_first_frame_boundary(self):
        """Test that the first frame is always predicted as a boundary."""
        output = self.predictor.forward(self.hidden)

        # First frame should always be a boundary (boundary_prob[:, 0, 1] should be 1.0)
        first_frame_boundary_prob = output.boundary_prob[:, 0, 1]
        expected = torch.ones(self.batch_size, device=self.device)
        torch.testing.assert_close(
            first_frame_boundary_prob,
            expected,
            rtol=1e-5, atol=1e-6
        )

        # First frame boundary mask should be True
        self.assertTrue(torch.all(output.boundary_mask[:, 0]))

    def test_compress_features(self):
        """Test feature compression functionality."""
        # Create a simple boundary mask
        boundary_mask = torch.tensor([
            [True, False, True, False, False],
            [True, True, False, True, False]
        ], device=self.device)

        # Create simple input features
        simple_hidden = torch.randn(2, 5, self.d_model, device=self.device)

        compressed = self.predictor._compress_features(
            simple_hidden, boundary_mask)

        # Check output shape
        max_boundaries = int(boundary_mask.sum(dim=-1).max())
        self.assertEqual(compressed.shape, (2, max_boundaries, self.d_model))

        # Check that compressed features are non-zero only where boundaries exist
        num_boundaries_per_batch = boundary_mask.sum(dim=-1)
        for batch_idx in range(2):
            num_boundaries = num_boundaries_per_batch[batch_idx]
            # Valid positions should have non-zero features (in most cases)
            valid_positions = compressed[batch_idx, :num_boundaries]
            # Invalid positions should be zero
            if num_boundaries < max_boundaries:
                invalid_positions = compressed[batch_idx, num_boundaries:]
                self.assertTrue(torch.allclose(
                    invalid_positions, torch.zeros_like(invalid_positions)))

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        # Create boundary masks with known ratios
        boundary_mask = torch.tensor([
            [True, False, True, False, False],  # 2/5 = 0.4
            [True, True, False, True, False]    # 3/5 = 0.6
        ], device=self.device)

        ratios = self.predictor.get_compression_ratio(boundary_mask)

        expected_ratios = torch.tensor([0.4, 0.6], device=self.device)
        torch.testing.assert_close(
            ratios, expected_ratios, rtol=1e-5, atol=1e-6)

    def test_different_input_sizes(self):
        """Test with different input sizes."""
        test_cases = [
            (1, 5, self.d_model),    # Small batch, short sequence
            (4, 20, self.d_model),   # Larger batch, longer sequence
            (1, 1, self.d_model),    # Single frame
        ]

        for batch_size, seq_len, d_model in test_cases:
            with self.subTest(batch_size=batch_size, seq_len=seq_len):
                hidden = torch.randn(batch_size, seq_len,
                                     d_model, device=self.device)
                output = self.predictor.forward(hidden)

                # Check shapes
                self.assertEqual(output.boundary_prob.shape,
                                 (batch_size, seq_len, 2))
                self.assertEqual(output.boundary_mask.shape,
                                 (batch_size, seq_len))
                self.assertEqual(output.selected_probs.shape,
                                 (batch_size, seq_len, 1))
                self.assertEqual(
                    output.compressed_features.shape[0], batch_size)
                self.assertEqual(output.compressed_features.shape[2], d_model)

    def test_edge_case_single_frame(self):
        """Test edge case with single frame input."""
        single_frame = torch.randn(1, 1, self.d_model, device=self.device)
        output = self.predictor.forward(single_frame)

        # With single frame, boundary probability should be 1.0 for boundary
        self.assertEqual(output.boundary_prob[0, 0, 1].item(), 1.0)
        self.assertTrue(output.boundary_mask[0, 0].item())

        # Compressed features should contain the single frame
        self.assertEqual(output.compressed_features.shape,
                         (1, 1, self.d_model))

    def test_edge_case_no_boundaries(self):
        """Test edge case where no boundaries are predicted (except first frame)."""
        # Create identical consecutive features to minimize boundary probability
        identical_features = torch.ones(1, 5, self.d_model, device=self.device)

        # Manually create boundary mask with only first frame as boundary
        boundary_mask = torch.tensor(
            [[True, False, False, False, False]], device=self.device)

        compressed = self.predictor._compress_features(
            identical_features, boundary_mask)

        # Should have only one boundary (the first frame)
        self.assertEqual(compressed.shape, (1, 1, self.d_model))

    def test_cosine_similarity_computation(self):
        """Test that cosine similarity is computed correctly."""
        # Create simple test case with known similarity
        test_hidden = torch.tensor([[[1.0, 0.0, 0.0],
                                   # Same as previous (cos_sim = 1.0)
                                     [1.0, 0.0, 0.0],
                                     # Orthogonal (cos_sim = 0.0)
                                     [0.0, 1.0, 0.0]]], device=self.device)

        # Create a simple predictor for 3D features
        test_predictor = HNetBoundaryPredictor(d_model=3, device=self.device)
        output = test_predictor.forward(test_hidden)

        # Check that boundary probabilities make sense
        # First frame should always be boundary (prob = 1.0)
        self.assertAlmostEqual(
            output.boundary_prob[0, 0, 1].item(), 1.0, places=5)

        # Second frame: identical features should have low boundary probability
        # cos_sim = 1.0 -> boundary_prob = (1-1)/2 = 0.0
        self.assertLess(output.boundary_prob[0, 1, 1].item(), 0.1)

        # Third frame: orthogonal features should have higher boundary probability
        # cos_sim = 0.0 -> boundary_prob = (1-0)/2 = 0.5
        self.assertGreater(output.boundary_prob[0, 2, 1].item(), 0.4)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        # Create a fresh input tensor with gradients enabled
        test_hidden = torch.randn(
            self.batch_size, self.seq_len, self.d_model,
            device=self.device, dtype=self.dtype, requires_grad=True
        )

        output = self.predictor.forward(test_hidden)

        # Create a more sensitive loss that ensures gradient flow
        # Use both boundary probabilities and compressed features
        # Focus on boundary probs
        boundary_loss = output.boundary_prob[:, :, 1].mean()
        feature_loss = output.compressed_features.mean()      # Use compressed features
        loss = boundary_loss + 0.1 * feature_loss
        loss.backward()

        # Check that gradients exist
        self.assertIsNotNone(test_hidden.grad)
        self.assertIsNotNone(self.predictor.q_proj_layer.weight.grad)
        self.assertIsNotNone(self.predictor.k_proj_layer.weight.grad)

        # Check that gradients are non-zero with a reasonable tolerance
        # Use a small tolerance to account for numerical precision
        self.assertFalse(torch.allclose(
            test_hidden.grad,
            torch.zeros_like(test_hidden.grad),
            atol=1e-8, rtol=1e-6
        ))

        # Also check that at least some gradients have meaningful magnitude
        grad_norm = torch.norm(test_hidden.grad)
        self.assertGreater(grad_norm.item(), 1e-6)

    def test_device_consistency(self):
        """Test that model works correctly on different devices."""
        # Test CPU (already tested in other methods)
        cpu_output = self.predictor.forward(self.hidden)

        # Test GPU if available
        if torch.cuda.is_available():
            gpu_predictor = HNetBoundaryPredictor(
                d_model=self.d_model,
                device=torch.device('cuda')
            )
            gpu_hidden = self.hidden.cuda()
            gpu_output = gpu_predictor.forward(gpu_hidden)

            # Check that outputs are on correct device
            self.assertEqual(gpu_output.boundary_prob.device.type, 'cuda')
            self.assertEqual(gpu_output.boundary_mask.device.type, 'cuda')
            self.assertEqual(
                gpu_output.compressed_features.device.type, 'cuda')


class TestWhisperBoundaryState(unittest.TestCase):
    """Test suite for WhisperBoundaryState dataclass."""

    def test_boundary_state_creation(self):
        """Test creation of WhisperBoundaryState."""
        batch_size = 2
        d_model = 512

        has_seen_features = torch.tensor([True, False])
        last_feature = torch.randn(batch_size, d_model)

        state = WhisperBoundaryState(
            has_seen_features=has_seen_features,
            last_feature=last_feature
        )

        self.assertTrue(torch.equal(
            state.has_seen_features, has_seen_features))
        self.assertTrue(torch.equal(state.last_feature, last_feature))


if __name__ == '__main__':
    # Set random seed for reproducible tests
    torch.manual_seed(42)

    # Run tests
    unittest.main(verbosity=2)
