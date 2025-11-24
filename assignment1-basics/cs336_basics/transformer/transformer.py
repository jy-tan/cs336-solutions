import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from cs336_basics.transformer.attention import CausalMultiHeadAttention
from cs336_basics.transformer.core import Embedding, Linear, RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadAttention(
            d_model, num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self, x: Float[Tensor, " ... seq_len d_model"], token_positions: Int[Tensor, " ... seq_len"] | None = None
    ) -> Float[Tensor, " ... seq_len d_model"]:
        """
        Forward pass for the Transformer block.

        FLOPs:
            - layer norm 1: 4 * batch_size * seq_len * d_model
            - attention: 8 * batch_size * seq_len * d_model^2
                + 6 * batch_size * seq_len * d_model
                + 4 * batch_size * seq_len^2 * d_model
                + 6 * batch_size * num_heads * seq_len^2
            - add: batch_size * seq_len * d_model
            - layer norm 2: 4 * batch_size * seq_len * d_model
            - feedforward (SwiGLU): 2 * 3 * batch_size * seq_len * d_model * d_ff
            - add: batch_size * seq_len * d_model

            Total: 8 * batch_size * seq_len * d_model^2
                + 16 * batch_size * seq_len * d_model
                + 4 * batch_size * seq_len^2 * d_model
                + 6 * batch_size * num_heads * seq_len^2
                + 6 * batch_size * seq_len * d_model * d_ff
        """
        x_orig = x.clone()

        # Multi-head attention sublayer
        x = self.ln1(x)
        x = self.attn(x, token_positions=token_positions) + x_orig

        # Feed-forward sublayer
        x_orig = x.clone()
        x = self.ln2(x)
        x = self.ffn(x) + x_orig

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Int[Tensor, " ... seq_len"]) -> Float[Tensor, " ... seq_len vocab_size"]:
        """
        Forward pass for the Transformer model.

        FLOPs:
            - transformer blocks: num_layers * (8 * batch_size * seq_len * d_model^2
                + 16 * batch_size * seq_len * d_model
                + 4 * batch_size * seq_len^2 * d_model
                + 6 * batch_size * num_heads * seq_len^2
                + 6 * batch_size * seq_len * d_model * d_ff)
            - final layer norm: 4 * batch_size * seq_len * d_model
            - linear: 2 * batch_size * seq_len * d_model * vocab_size

            Total: num_layers * (8 * batch_size * seq_len * d_model^2
                + 16 * batch_size * seq_len * d_model
                + 4 * batch_size * seq_len^2 * d_model
                + 6 * batch_size * num_heads * seq_len^2
                + 6 * batch_size * seq_len * d_model * d_ff)
                + 4 * batch_size * seq_len * d_model + 2 * batch_size * seq_len * d_model * vocab_size
        """
        x: Float[Tensor, " ... seq_len d_model"] = self.token_embeddings(token_ids)

        batch_shape = token_ids.shape[:-1]
        seq_len = token_ids.shape[-1]
        token_positions = torch.arange(seq_len, device=token_ids.device)

        # Expand to match batch dims
        for _ in range(len(batch_shape)):
            token_positions = token_positions.unsqueeze(0)
        token_positions = token_positions.expand(*batch_shape, seq_len)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits
