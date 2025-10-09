import torch


def final(foo, temperature: float = 10.0):
    print("[final] input foo shape:", foo.shape)
    print("[final] temperature:", temperature)

    # Soft mask that heavily favors foo == 0 but keeps gradients alive elsewhere
    weights = torch.exp(-temperature * foo.abs())
    print("[final] raw soft weights:", weights)

    denominator = weights.sum(dim=1, keepdim=True) + 1e-9
    print("[final] denominator (sum over dim=1):", denominator)

    normalized = weights / denominator
    print("[final] normalized soft weights:", normalized)

    return normalized


def common(boundaries):
    print("[common] raw boundaries:", boundaries)
    boundaries = boundaries.clone()
    print("[common] cloned boundaries:", boundaries)

    n_segments = boundaries.sum(dim=-1).max().item()
    print("[common] n_segments:", n_segments)

    if n_segments == 0:
        print("[common] no segments found, returning None")
        return None

    tmp = torch.zeros_like(boundaries).unsqueeze(2) + torch.arange(
        start=0,
        end=n_segments,
        device=boundaries.device
    )
    print("[common] tmp shape:", tmp.shape)
    print("[common] tmp sample:", tmp)

    hh1 = boundaries.cumsum(1)
    print("[common] cumulative sum hh1:", hh1)

    hh1 -= boundaries
    print("[common] hh1 after subtracting boundaries:", hh1)

    foo = tmp - hh1.unsqueeze(-1)
    print("[common] foo shape:", foo.shape)
    print("[common] foo sample:", foo)

    return foo


def downsample(boundaries, hidden):
    print("[downsample] boundaries shape:", boundaries.shape)
    print("[downsample] hidden shape:", hidden.shape)

    input_dtype = hidden.dtype
    print("[downsample] input dtype:", input_dtype)

    foo = common(boundaries)
    print("[downsample] result of common (foo):", foo)

    if foo is None:
        print("[downsample] foo is None, returning empty tensor")
        return torch.empty(0, hidden.size(1), hidden.size(2), device=hidden.device, dtype=input_dtype)
    else:
        bar = final(foo=foo)
        print("[downsample] bar (after final):", bar)

        bar = bar.to(dtype=input_dtype)
        print("[downsample] bar dtype after cast:", bar.dtype)

        shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, bar)
        print("[downsample] shortened_hidden shape:", shortened_hidden.shape)
        print("[downsample] shortened_hidden tensor:", shortened_hidden)

        return shortened_hidden


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)

    # Simple 1-batch example
    batch_size = 1
    seq_length = 5
    hidden_dim = 3

    # Boundaries tensor with a start token (0) and two boundaries set to 1
    boundaries = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.float32)
    print("[main] boundaries tensor:", boundaries)

    # Hidden tensor shaped (L, B, D)
    hidden = torch.arange(seq_length * batch_size * hidden_dim, dtype=torch.float32).reshape(seq_length, batch_size, hidden_dim)
    print("[main] hidden tensor:", hidden)

    shortened_hidden = downsample(boundaries, hidden)
    print("[main] final shortened_hidden:", shortened_hidden)

    print("\n[main] checking gradient flow w.r.t. boundaries")
    boundaries_grad = boundaries.clone().detach().requires_grad_(True)
    try:
        shortened_hidden_grad = downsample(boundaries_grad, hidden)
        loss = shortened_hidden_grad.sum()
        print("[main] loss from shortened_hidden_grad.sum():", loss)
        loss.backward()
        print("[main] boundaries_grad.grad:", boundaries_grad.grad)
    except RuntimeError as exc:
        print("[main] encountered RuntimeError while checking gradients:", exc)

    print("\n[main] suboptimal boundary fine-tuning demo")
    torch.manual_seed(0)

    # Hidden states with two obvious clusters but intentionally poor boundary placement
    hidden_demo = torch.tensor([
        [0.0],  # timeslice 0
        [2.0],  # timeslice 1
        [4.0],  # timeslice 2
        [10.0],  # timeslice 3
        [12.0],  # timeslice 4
    ], dtype=torch.float32).reshape(5, 1, 1)
    print("[demo] hidden_demo:", hidden_demo.squeeze(-1).squeeze(-1))

    # Boundaries start two segments at indices 1 and 3 (second segment misses the last value we care about)
    boundaries_demo = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]], dtype=torch.float32, requires_grad=True)
    print("[demo] initial boundaries_demo:", boundaries_demo.detach())

    target_segments = torch.tensor([[[1.0]], [[11.0]]])
    print("[demo] target shortened representation:", target_segments.squeeze(-1).squeeze(-1))

    shortened_demo = downsample(boundaries_demo, hidden_demo)
    print("[demo] shortened_demo output:", shortened_demo.squeeze(-1).squeeze(-1))

    demo_loss = torch.nn.functional.mse_loss(shortened_demo, target_segments)
    print("[demo] loss before gradient step:", demo_loss.item())

    demo_loss.backward()
    print("[demo] boundary gradients:", boundaries_demo.grad)

    # Attempt a tiny gradient step while renormalizing to keep roughly two segments
    step_size = 0.05
    with torch.no_grad():
        updated_boundaries = boundaries_demo - step_size * boundaries_demo.grad
        updated_boundaries.clamp_(min=0.0)
        updated_boundaries[0, 0] = 0.0  # keep first token fixed at zero

        original_sum = boundaries_demo.detach().sum()
        updated_sum = updated_boundaries.sum()
        if updated_sum > 0:
            scale = (original_sum / updated_sum).item()
            updated_boundaries.mul_(scale)

    print("[demo] renormalized updated_boundaries:", updated_boundaries.detach())

    try:
        shortened_after = downsample(updated_boundaries.detach(), hidden_demo)
        new_loss = torch.nn.functional.mse_loss(shortened_after, target_segments)
        print("[demo] shortened_after output:", shortened_after.squeeze(-1).squeeze(-1))
        print("[demo] loss after naive gradient step:", new_loss.item())
    except RuntimeError as exc:
        print("[demo] downsample failed after update with RuntimeError:", exc)
