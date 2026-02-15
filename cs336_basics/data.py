import numpy as np
import torch


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    x_t = torch.as_tensor(x)

    # Valid start indices t satisfy: t + context_length < n  =>  t in [0, n-context_length-1]
    # torch.randint uses an exclusive high bound.
    starts = torch.randint(0, n - context_length, (batch_size,), dtype=torch.long)

    offsets = torch.arange(context_length, dtype=torch.long).unsqueeze(0)  # (1, m)
    idx = starts.unsqueeze(1) + offsets  # (B, m)

    inputs = x_t[idx]
    targets = x_t[idx + 1]

    # Move to requested device
    inputs = inputs.to(device)
    targets = targets.to(device)

    return inputs, targets
