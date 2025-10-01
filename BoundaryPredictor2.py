import torch.nn.utils.rnn
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loss import binomial_loss, hinge_loss
from utils import delete, downsample


class BoundaryPredictor2(nn.Module):
    def __init__(self, input_dim, hidden_dim, prior, temp=1, threshold=0.5):
        """
        input_dim: dimensionality of per-token vectors (D)
        tau: Gumbel-Sigmoid temperature
        """
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.threshold = threshold

        # Linear projections for computing similarity (initialized as identity)
        self.q_proj_layer = nn.Linear(
            input_dim, input_dim, bias=False)
        self.k_proj_layer = nn.Linear(
            input_dim, input_dim, bias=False)

        # Initialize as identity matrices (same as RoutingModule)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(input_dim))
            self.k_proj_layer.weight.copy_(torch.eye(input_dim))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def set_prior(self, prior):
        self.prior = prior

    def forward(self, hidden, attention_mask=None):
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            # Move normalization to before the projection layer
            self.q_proj_layer(F.normalize(hidden[:, :-1])),
            self.k_proj_layer(F.normalize(hidden[:, 1:]))
        )

        probs = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        probs = F.pad(probs, (1, 0), "constant", 1.0)

        bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=self.temp,
            probs=probs,
        )

        soft_boundaries = bernoulli.rsample()

        hard_boundaries = (soft_boundaries > self.threshold).float()
        hard_boundaries = (
            hard_boundaries - soft_boundaries.detach() + soft_boundaries
        )

        # Apply attention mask to hard boundaries to ensure masked positions are not boundaries
        if attention_mask is not None:
            hard_boundaries = hard_boundaries * attention_mask

            # Ensure we always close out the final real segment by forcing a boundary
            # at the first padded position (if any). This keeps every non-padded
            # frame while preventing padded timesteps from leaking through.
            pad_mask = attention_mask == 0
            if pad_mask.any():
                first_pad_mask = pad_mask & (
                    pad_mask.long().cumsum(dim=1) == 1)
                # Shift the boundary one position to the left (to the last real token)
                last_real_mask = torch.roll(first_pad_mask, shifts=-1, dims=1)
                # Clear the last column to avoid wrapping
                last_real_mask[:, -1] = False
                hard_boundaries = torch.maximum(
                    hard_boundaries, last_real_mask.float()
                )

        pooled = downsample(hard_boundaries, hidden)  # S x B x D
        # pooled = delete(hard_boundaries, hidden)  # S x B x D

        pooled = pooled.transpose(0, 1)

        shortened_attention_mask = None

        if attention_mask is not None:
            keep_mask = hard_boundaries == 1
            batch_size = attention_mask.shape[0]
            shortened_masks = []

            for b in range(batch_size):
                keep_indices = keep_mask[b].nonzero(as_tuple=True)[0]

                # Extract the relevant positions from 1D attention mask
                original_mask = attention_mask[b]
                shortened_mask = original_mask[keep_indices]
                shortened_masks.append(shortened_mask)

            boundary_counts = hard_boundaries.sum(dim=1)

            max_len = max((mask.size(0)
                          for mask in shortened_masks), default=0)
            expected_max_len = int(
                boundary_counts.max().item()) if batch_size > 0 else 0
            if max_len != expected_max_len:
                print(
                    "[BoundaryPredictor2] Max boundary count mismatch:",
                    f"expected_max_len={expected_max_len}",
                    f"observed_max_len={max_len}",
                    "boundary_counts=", boundary_counts.detach().cpu()
                )
                raise ValueError(
                    "shortened_attention_mask length mismatch detected.")

            # Automatically pad to max length and stack
            if max_len == 0:
                shortened_attention_mask = attention_mask.new_zeros(
                    (batch_size, 0))
            else:
                shortened_attention_mask = torch.nn.utils.rnn.pad_sequence(
                    shortened_masks, batch_first=True, padding_value=0.0)

                unique_vals = torch.unique(shortened_attention_mask)
                if not torch.all((unique_vals == 0.0) | (unique_vals == 1.0)):
                    print(
                        "[BoundaryPredictor2] Non-binary shortened_attention_mask values detected:",
                        unique_vals.detach().cpu()
                    )
                    raise ValueError(
                        "shortened_attention_mask must be binary (0s and 1s)."
                    )

                if shortened_attention_mask.size(1) > 1:
                    if not torch.all(
                        shortened_attention_mask[:, :-
                                                 1] >= shortened_attention_mask[:, 1:]
                    ):
                        print(
                            "[BoundaryPredictor2] Non-monotonic shortened_attention_mask rows:",
                            shortened_attention_mask.detach().cpu()
                        )
                        raise ValueError(
                            "shortened_attention_mask rows must be left-monotonic (ones then zeros)."
                        )

            expected_mask = (
                torch.arange(max_len, device=shortened_attention_mask.device)
                .unsqueeze(0) < boundary_counts.unsqueeze(1)
            ).float()
            if not torch.equal(shortened_attention_mask, expected_mask):
                print(
                    "[BoundaryPredictor2] shortened_attention_mask does not match expected pattern.",
                    "expected_mask=", expected_mask.detach().cpu(),
                    "actual_mask=", shortened_attention_mask.detach().cpu(),
                    "boundary_counts=", boundary_counts.detach().cpu()
                )
                raise ValueError(
                    "shortened_attention_mask structure mismatch detected."
                )

            kept_counts = shortened_attention_mask.sum(dim=1)
            if not torch.all(kept_counts.long() == boundary_counts.long()):
                print(
                    "[BoundaryPredictor2] Kept count mismatch: kept_counts=", kept_counts.detach(
                    ).cpu(),
                    "boundary_counts=", boundary_counts.detach().cpu()
                )
                raise ValueError(
                    "shortened_attention_mask count mismatch detected."
                )

            if pooled.shape[0] != shortened_attention_mask.shape[0]:
                print(
                    "[BoundaryPredictor2] pooled batch dim:", pooled.shape[0],
                    "shortened_attention_mask batch dim:", shortened_attention_mask.shape[0]
                )
                raise ValueError(
                    "Pooled hidden state batch dimension does not match shortened attention mask."
                )

            if pooled.shape[1] != shortened_attention_mask.shape[1]:
                print(
                    "[BoundaryPredictor2] pooled sequence length:", pooled.shape[1],
                    "shortened_attention_mask sequence length:", shortened_attention_mask.shape[
                        1]
                )
                raise ValueError(
                    "Pooled hidden state sequence length does not match shortened attention mask."
                )

        num_boundaries_tensor = hard_boundaries.sum()
        if attention_mask is not None:
            total_positions_tensor = attention_mask.sum()
        else:
            total_positions_tensor = torch.tensor(
                hard_boundaries.numel(), device=hard_boundaries.device, dtype=torch.float)

        loss = self.calc_loss(num_boundaries_tensor, total_positions_tensor)
        self.last_loss = loss  # Store the calculated loss

        # Convert to scalars for metrics (after loss calculation)
        num_boundaries = num_boundaries_tensor.item()
        total_positions = total_positions_tensor.item()

        return pooled, loss, num_boundaries, total_positions, shortened_attention_mask

    def calc_loss(self, num_boundaries, total_positions):
        return binomial_loss(num_boundaries, total_positions, self.prior) / (64 ** 2)
        # return hinge_loss(num_boundaries, total_positions, self.prior, .03)
