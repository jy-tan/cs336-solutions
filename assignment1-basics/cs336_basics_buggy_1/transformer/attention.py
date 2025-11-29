import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

import torch.nn.functional as F

from cs336_basics_buggy_1.transformer.core import Linear, RotaryPositionalEmbedding, softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Scaled dot product attention.

    Args:
        Q: Projected query tensor, shape: (batch, seq_len, d_k)
        K: Projected key tensor, shape: (batch, seq_len, d_k)
        V: Projected value tensor, shape: (batch, seq_len, d_v)
        mask: Boolean matrix of shape (seq_len, seq_len)

    Returns:
        Float[Tensor, " ... queries d_v"]: Output tensor

    FLOPs:
        - dot product: 2 * batch_size * seq_len^2 * d_k
        - scale: batch_size * seq_len^2
        - softmax: 5 * batch_size * seq_len^2
        - einsum: 2 * batch_size * seq_len^2 * d_v

        Total: batch_size * seq_len^2 * (2 * d_k + 2 * d_v + 6)

        Scaled dot product attention is expensive at long sequences.
    """
    d_k = Q.size(-1)
    # Note: in self-attention, queries = keys = sequence length
    # V has shape " ... keys d_v" bc there is 1 value per key, derived from the input sequence.

    qk_dot_product = einsum(Q, K, " ... q d_k, ... k d_k -> ... q k")
    # "How well does each query match each key?"

    # scaled_qk_dot_product = qk_dot_product * (d_k**0.5)  # BUG: should be divide by sqrt(d_k)
    scaled_qk_dot_product = qk_dot_product / (d_k ** 0.5)

    # Mask before softmax
    if mask is not None:
        scaled_qk_dot_product = scaled_qk_dot_product.masked_fill(~mask, -float("inf"))

    # attention_weights = softmax(scaled_qk_dot_product, dimension=-2)  # BUG: softmax on the last dimension (keys)
    attention_weights = softmax(scaled_qk_dot_product, dimension=-1)
    # "For each query, what proportion of each key?"

    output = einsum(attention_weights, V, " ... q k, ... k d_v -> ... q d_v")
    # "For each query, combine values according to weights"

    return output


class CausalMultiHeadAttention(nn.Module):
    # TOOO: a bunch of duplication here, ideally combine q, k, v into a single matrix

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        if d_model % num_heads != 0:
            raise ValueError("d_model is not divisible by num_heads")

        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(self.d_k, theta=theta, max_seq_len=max_seq_len, device=device)

        # Lower triangle is 1, including diagonals
        # True = pass through, False = mask
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, " ... seq_len d_model"]:
        """
        Forward pass for causal multi-head attention

        FLOPs:
            - q_proj: 2 * batch_size * seq_len * d_model^2
            - k_proj: 2 * batch_size * seq_len * d_model^2
            - v_proj: 2 * batch_size * seq_len * d_model^2
            - Q rope: batch_size * num_heads * seq_len * d_k // 2 * 6
            - V rope: batch_size * num_heads * seq_len * d_k // 2 * 6
            - scaled dot product attn: batch_size * num_heads * seq_len^2 * (2 * d_k + 2 * d_v + 6)
            - output_proj: 2 * batch_size * seq_len * d_model^2

            Total: 8 * batch_size * seq_len * d_model^2
                + 6 * batch_size * num_heads * seq_len * d_k
                + batch_size * num_heads * seq_len^2 * (2 * d_k + 2 * d_v + 6)

            Simplifying (d_k = d_v = d_model / num_heads):
                8 * batch_size * seq_len * d_model^2
                + 6 * batch_size * seq_len * d_model
                + 4 * batch_size * seq_len^2 * d_model
                + 6 * batch_size * num_heads * seq_len^2

            Note that the dominant term is typically 8 * batch_size * seq_len * d_model^2
            assuming seq_len << d_model.
        """

        seq_len = x.shape[-2]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads
        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)

        if token_positions is not None:
            Q = self.rope(Q, token_positions=token_positions)
            K = self.rope(K, token_positions=token_positions)
            V = self.rope(V, token_positions=token_positions) # BUG: shouldn't be applied to V

        mask = self.causal_mask[:seq_len, :seq_len]

        attention_output = scaled_dot_product_attention(Q, K, V, mask)
        attention_output = rearrange(attention_output, " ... num_heads seq_len d_k -> ... seq_len (num_heads d_k)")

        # BUG: missing softmax here

        output = self.output_proj(attention_output)
        return output


if __name__ == "__main__":
    Q = torch.randn(2, 4, 8, 16)
    K = torch.randn(2, 4, 8, 16)
    V = torch.randn(2, 4, 8, 16)

    actual = scaled_dot_product_attention(Q, K, V)
    expected = F.scaled_dot_product_attention(Q, K, V)

    print(f"Actual shape: {actual.shape}")
    print(f"Expected shape: {expected.shape}")
    
    print(f"Actual: {actual}")
    print(f"Expected: {expected}")

    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-8)
