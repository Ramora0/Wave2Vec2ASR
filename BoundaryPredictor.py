import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# def downsample(boundaries, hidden, new=False):
#     """
#     Downsamples hidden states based on boundaries using simple loops and averaging.
#     A new segment starts at index 0 and at every index t where boundaries[b, t] == 1.

#     Args:
#         boundaries (torch.Tensor): Tensor of shape (B, L) where 1 indicates
#                                    the start of a new segment *after* the current position.
#         hidden (torch.Tensor): Tensor of shape (B, L, D) containing hidden states.

#     Returns:
#         torch.Tensor: Pooled hidden states of shape (S, B, D), where S is the
#                       maximum number of segments in the batch. Padded with zeros.
#     """
#     batch_size, seq_len, model_dim = hidden.shape
#     device = hidden.device

#     pooled_batch = []
#     max_segments = 0

#     for b in range(batch_size):
#         b_boundaries = boundaries[b] # Shape (L,)
#         b_hidden = hidden[b]         # Shape (L, D)

#         pooled_segments = []
#         current_segment_start_idx = 0

#         # Iterate through potential boundary points
#         for t in range(seq_len-1):
#             if b_boundaries[t] == 1:
#                 segment = b_hidden[current_segment_start_idx:t+1]
#                 pooled_segments.append(segment.mean(dim=0))

#                 # New segment starts at index t
#                 current_segment_start_idx = t+1

#         # Add the last segment (from last boundary or from start if no boundaries)
#         last_segment = b_hidden[current_segment_start_idx:seq_len]
#         pooled_segments.append(last_segment.mean(dim=0))

#         b_pooled = torch.stack(pooled_segments) # Shape (num_segments, D)
#         pooled_batch.append(b_pooled)
#         max_segments = max(max_segments, b_pooled.shape[0])

#     pooled_tensor = torch.nn.utils.rnn.pad_sequence(pooled_batch, batch_first=False, padding_value=0.0)

#     return pooled_tensor

def downsample(boundaries, hidden, pooling="mean"):
    """
    Downsamples hidden states based on boundaries using vectorized torch operations.
    A new segment starts at index 0 and at every index t+1 where boundaries[b, t] == 1.

    Args:
        boundaries (torch.Tensor): Tensor of shape (B, L), dtype=float or long.
                                   1 indicates the *end* of a segment at index t.
        hidden (torch.Tensor): Tensor of shape (B, L, D) containing hidden states.

    Returns:
        torch.Tensor: Pooled hidden states of shape (S, B, D), where S is the
                      maximum number of segments in the batch. Padded with zeros.
    """
    batch_size, seq_len, model_dim = hidden.shape
    device = hidden.device
    dtype = hidden.dtype

    # Ensure boundaries is long type for indexing and cumsum
    boundaries = boundaries.long()

    # 1. Calculate segment IDs for each time step t in each batch item b.
    # segment_ids[b, t] = ID of the segment that hidden[b, t] belongs to.
    # A new segment starts at t=0 and at t+1 if boundaries[b, t] == 1.
    # We use cumsum on boundaries shifted right by one, prepended with 0.
    segment_ids = torch.cat([
        torch.zeros_like(boundaries[:, :1]),  # Segment 0 starts at index 0
        # Boundary at t-1 means new segment starts at t
        boundaries[:, :-1]
    ], dim=1).cumsum(dim=1)  # Shape: (B, L)

    # 2. Determine the number of segments for each batch item and the maximum.
    num_segments_per_batch = segment_ids[:, -1] + 1  # Max segment ID + 1
    max_segments = num_segments_per_batch.max(
    ).item() if batch_size > 0 and seq_len > 0 else 0

    if pooling == "mean":
        # 3. Sum hidden states per segment using scatter_add_.
        # Target tensor for sums: (B, max_segments, D)
        summed_hidden = torch.zeros(
            batch_size, max_segments, model_dim, device=device, dtype=dtype)
        # Index for scatter_add_: needs to match shape of source (hidden) along scatter dim (1)
        # Index shape: (B, L) -> expand to (B, L, D)
        index = segment_ids.unsqueeze(-1).expand_as(hidden)  # Shape: (B, L, D)
        summed_hidden.scatter_add_(dim=1, index=index, src=hidden)

        # 4. Count segment lengths using scatter_add_.
        # Target tensor for counts: (B, max_segments)
        segment_lengths = torch.zeros(
            batch_size, max_segments, device=device, dtype=torch.int64)
        # Index for scatter_add_: (B, L)
        # Source for scatter_add_: ones(B, L)
        segment_lengths.scatter_add_(
            dim=1, index=segment_ids, src=torch.ones_like(segment_ids))

        # 5. Calculate the mean.
        # Avoid division by zero for potentially empty segments (scatter_add initializes with 0).
        # Clamp segment_lengths to minimum 1 before dividing.
        # Shape: (B, max_segments, 1)
        segment_lengths_clamped = segment_lengths.unsqueeze(-1).clamp(min=1)
        # Shape: (B, max_segments, D)
        pooled_batch = summed_hidden / segment_lengths_clamped
    elif pooling == "max":
        index = segment_ids.unsqueeze(-1).expand_as(hidden)  # Shape: (B, L, D)

        pooled_batch = torch.scatter_reduce(
            input=torch.full((batch_size, max_segments, model_dim), torch.finfo(
                dtype).min, device=device, dtype=dtype),  # Initialize with min value
            dim=1,
            index=index,
            src=hidden,  # Use src argument for the source tensor
            reduce="amax",
            include_self=False  # Exclude initial min values from reduction
        )

    # 6. Transpose to match the expected output shape (S, B, D).
    pooled_tensor = pooled_batch.transpose(0, 1)  # Shape: (max_segments, B, D)

    return pooled_tensor


class BoundaryPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, prior, temp=1, threshold=0.5):
        """
        input_dim: dimensionality of per-token vectors (D)
        hidden_dim: hidden size of the MLP
        tau: Gumbel-Sigmoid temperature
        """
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.threshold = threshold

        hidden = hidden_dim
        self.boundary_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def set_prior(self, prior):
        self.prior = prior

    def forward(self, hidden):
        # print("Hidden", hidden.shape)
        bs, seq_len, model_dim = hidden.shape
        logits = self.boundary_mlp(
            hidden).squeeze(-1)
        probs = torch.sigmoid(logits)

        bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=self.temp,
            probs=probs,
        )

        soft_boundaries = bernoulli.rsample()

        hard_boundaries = (soft_boundaries > self.threshold).float()
        hard_boundaries = (
            hard_boundaries - soft_boundaries.detach() + soft_boundaries
        )

        pooled = downsample(
            hard_boundaries, hidden  # , hidden.new_zeros((1, bs, model_dim))
        )  # S x B x D
        pooled = pooled.transpose(0, 1)

        loss = self.calc_loss(hard_boundaries)
        self.last_loss = loss  # Store the calculated loss

        return pooled, loss

    def calc_loss(self, preds):
        binomial = torch.distributions.binomial.Binomial(
            preds.size(-1),
            probs=torch.Tensor([self.prior]).to(preds.device)
        )
        loss_boundaries = -binomial.log_prob(
            preds.sum(dim=-1)
        ).mean() / preds.size(-1)

        return loss_boundaries


if __name__ == "__main__":
    # Simple downsampling test
    hidden = torch.randn(1, 5, 2)
    print(f"Hidden: {hidden[0]}")
    boundaries = torch.tensor([[0, 1, 0, 1, 1]])

    pooled = downsample(boundaries, hidden)
    print("Pooled shape:", pooled.shape)  # Expected shape: (S, B, D)
    print("Pooled tensor:", pooled)

    # device = torch.device("cuda")

    # total_time_old = 0
    # total_time_new = 0
    # num_iterations = 100

    # for i in tqdm(range(num_iterations)):
    #     hidden = torch.randn(16, 160, 512, device=device)
    #     boundaries = torch.randint(0, 2, (16, 160), device=device)

    #     start_time = time.time()
    #     pooled_1 = downsample(boundaries, hidden)
    #     total_time_old += (time.time() - start_time)

    #     start_time = time.time()
    #     pooled_2 = downsample_fast(boundaries, hidden)
    #     total_time_new += (time.time() - start_time)

    #     if not torch.allclose(pooled_1, pooled_2, atol=1e-6):
    #         print(f"Iteration {i}: Pooled tensors are not close!")

    # print(f"\nAverage time for old function: {total_time_old / num_iterations:.6f} seconds")
    # print(f"Average time for new function: {total_time_new / num_iterations:.6f} seconds")

"""
v0 (loops):       215.4
v1 (new padding): 213.6
"""
