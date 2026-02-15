import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x_float = x.to(torch.float32)
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x_float * rms

        return (x_norm * self.weight).to(in_dtype)
