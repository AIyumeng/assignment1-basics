import torch
import torch.nn as nn
from .modules import Linear, Embedding, MHA, RMSNorm, SwiGLU


class Transformer_Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool = False,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rope = use_rope
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.attn = MHA(d_model, num_heads, use_rope, theta, max_seq_len)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.swiglu = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_copy = x
        x = self.ln1(x)
        token_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        x = self.attn(x, token_positions=token_positions)
        x = x + x_copy

        x_copy = x
        x = self.ln2(x)
        x = self.swiglu(x)
        x = x + x_copy

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rope = use_rope
        self.rope_theta = rope_theta

        self.embedding = Embedding(vocab_size, d_model)

        self.transformer = nn.ModuleList(
            [
                Transformer_Block(
                    d_model, num_heads, d_ff, use_rope, rope_theta, context_length
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for i in range(self.num_layers):
            x = self.transformer[i](x)

        x = self.ln_final(x)
        x = self.lm_head(x)

        return x
