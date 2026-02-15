import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self._init_weight()

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.weight[x]

    def _init_weight(self):
        mean = 0.0
        std = 1.0
        torch.nn.init.trunc_normal_(
            self.weight, mean=mean, std=std, a=-3 * std, b=3 * std
        )
