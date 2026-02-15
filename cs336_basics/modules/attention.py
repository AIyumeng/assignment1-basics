import torch
import torch.nn as nn
from .linear import Linear
from .pe import RoPE

def softmax(
    logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    max_logits = torch.max(logits, dim=dim, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    sum_exp_logits = torch.sum(exp_logits, dim=dim, keepdim=True)
    return exp_logits / sum_exp_logits


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = q.shape[-1]
    attn_logits = torch.einsum("... i d, ... j d -> ... i j", q, k) / (d_k**0.5)

    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))

    attn_weights = softmax(attn_logits, dim=-1)
    output = torch.einsum("... i j, ... j d -> ... i d", attn_weights, v)
    return output


class MHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        rope_theta: float|None = None,
        max_seq_len: int|None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        if self.use_rope:
            assert rope_theta is not None and max_seq_len is not None, "RoPE requires rope_theta and max_seq_len"
            self.pe = RoPE(rope_theta, self.d_k, max_seq_len, device=device)
        # Define the projection layers for Q, K, V
        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project the input to Q, K, V
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if self.use_rope and token_positions is not None:
            q = self.pe(q, token_positions)
            k = self.pe(k, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool().unsqueeze(0).unsqueeze(0)

        # Compute attention for each head and concatenate the results
        attn_output = attention(q, k, v, mask)

        # Project the output back to the original dimension
        output = self.w_o(attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model))
        return output