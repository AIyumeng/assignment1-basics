import einops
import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_k % 2 == 0

        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # [d_k/2]
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        )

        # [max_seq_len]
        positions = torch.arange(max_seq_len, device=device)

        # [max_seq_len, d_k/2]
        theta_mat = torch.einsum("i,j->ij", positions, inv_freq)

        # 👉 注意：这里先算，再 register
        cos = torch.cos(theta_mat)
        sin = torch.sin(theta_mat)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _rotate_half(self, x):
        x = einops.rearrange(x, "... (d j) -> ... d j", j=2)
        x1, x2 = x.unbind(dim=-1)
        return einops.rearrange(
            torch.stack((-x2, x1), dim=-1),
            "... d j -> ... (d j)",
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        """
        x: [..., seq_len, d_k]
        token_positions: [..., seq_len]
        """
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.unsqueeze(0)

        # [ ..., seq_len, d_k/2 ]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # 扩展到偶/奇维
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

        return x * cos + self._rotate_half(x) * sin