import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

    def _init_weight(self):
        mean = 0.0
        std = 1.0 / (2 * (self.in_features + self.out_features) ** 0.5)
        torch.nn.init.trunc_normal_(
            self.weight, mean=mean, std=std, a=-3 * std, b=3 * std
        )
