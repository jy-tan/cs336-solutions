import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Int
from torch import Tensor


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # y = Wx.
        # x is a column vector of shape (in_features, 1).
        # W is a matrix of shape (out_features, in_features).
        # y is a column vector of shape (out_features, 1).
        self.weight: Float[Tensor, " out_features in_features"] = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        sigma = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: Float[Tensor, " ... in_features"]) -> Float[Tensor, " ... out_features"]:
        """
        Forward pass for Linear.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)

        FLOPs:
            2 * batch_size * seq_len * in_features * out_features
        """
        return einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct an Embedding layer from a given number of embeddings and embedding dimension.

        Args:
            num_embeddings (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors (i.e., d_model).
            device (torch.device | str | None): Device to store the parameters on.
            dtype (torch.dtype | None): Data type of the parameters.

        FLOPs:
            None (just a lookup)
        """
        super().__init__()
        self.weight: Float[Tensor, " num_embeddings embedding_dim"] = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        sigma = 1
        nn.init.trunc_normal_(self.weight, std=sigma, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... embedding_dim"]:
        # Select rows from the weight matrix corresponding to the token ids.
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight: Float[Tensor, d_model] = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        """
        Forward pass for RMSNorm.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of shape (..., d_model)

        FLOPs:
            4 * batch_size * seq_len * d_model
            (each token goes through rsqrt, square, mean, multiple by RMS inverse, multiply by weight)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Pure pytorch implementation
        # result = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        # einops implementation
        rms_inv = torch.rsqrt(reduce(x**2, "... d_model -> ... 1", "mean")) + self.eps
        result = x * rms_inv * self.weight

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        SwiGLU module.
        Formula: SwiGLU(x) = (SiLU(xW1^T) ⊙ xW3^T)W2^T

        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension (typically ~8/3 * d_model, multiple of 64)
            device: Device to store parameters
            dtype: Data type of parameters
        """
        super().__init__()

        # TODO: do we need to scale d_ff here or can we assume it's already scaled by the caller?

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        """
        Forward pass for SwiGLU.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of shape (..., d_model)

        FLOPs:
            2 * 3 * batch_size * seq_len * d_model * d_ff
        """
        # SiLU
        gate = self.w1(x)
        gate = gate * torch.sigmoid(gate)

        # xW3^T
        value = self.w3(x)

        hidden = gate * value
        result = self.w2(hidden)

        return result


# TODO: understand RoPE better


def get_cos_sin(
    max_seq_len: int, theta_base: float, d_k: int, device
) -> tuple[Float[Tensor, "max_seq d_k // 2"], Float[Tensor, "max_seq d_k // 2"]]:
    """
    Get cos and sin for every position
    """
    # for i = 1 (second sequence)
    thetas = torch.tensor(theta_base, device=device).unsqueeze(0).repeat(d_k // 2)
    j = torch.arange(0, d_k // 2, device=device)
    inv_freqs: Float[Tensor, d_k // 2] = theta_base ** (-2 * j / d_k)
    thetas: Float[Tensor, max_seq_len, d_k // 2] = torch.outer(torch.arange(max_seq_len, device=device), inv_freqs)
    return thetas.cos(), thetas.sin()


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_k: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device: torch.device | str | None = None,
    ):
        """
        Construct the RoPE module with precomputed cos/sin buffers.

        Args:
            d_k: Dimension of query/key vectors (must be even)
            theta: Base for computing rotation frequencies (Θ)
            max_seq_len: Maximum sequence length to precompute
            device: Device to store buffers on
        """
        super().__init__()
        cos, sin = get_cos_sin(max_seq_len, theta, d_k, device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_k"],
        token_positions: Int[Tensor, " ... seq_len"],
    ) -> Float[Tensor, " ... seq_len d_k"]:
        """
        Apply RoPE to input tensor.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Position indices of shape (..., seq_len)

        Returns:
            Rotated tensor of same shape

        FLOPs:
            batch_size * seq_len * d_k // 2 * 6
            (4 multiplies + 2 adds per element)
        """
        in_dtype = x.dtype
        cos: Float[Tensor, "seq_len d_k // 2"] = self.cos[token_positions]
        sin: Float[Tensor, "seq_len d_k // 2"] = self.sin[token_positions]

        x_pairs = rearrange(x.to(torch.float32), "... seq_len (d_k_half t) -> ... seq_len d_k_half t", t=2)
        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]

        row1 = x1 * cos - x2 * sin
        row2 = x2 * sin + x1 * cos
        rotated = torch.stack([row1, row2], dim=-1)
        rotated = rearrange(rotated, "... seq_len d_k_half t -> ... seq_len (d_k_half t)", t=2)
        return rotated.to(in_dtype)


def softmax(x: Tensor, dimension: int):
    """
    Softmax function (with numerical stability).

    $$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}$$

    where $c$ is any constant. Here, we choose the constant to be max(x) along the chosen dimension.

    This means that the largest value will be $e^{x_{\max} - x_{\max}} = 1$.
    All other values will be $e^{x_i - x_{\max}} < 1$.

    Args:
        x: Input tensor
        dimension: Dimension to apply softmax to

    Returns:
        Softmax of the input tensor

    FLOPs:
        5 * batch_size * seq_len * D
        (batch_size * seq_len for each of {find max, subtract max, exp, sum, divide}
        and D is the size of the dimension that the softmax is operating over)
    """

    # Get the max along the dimension
    x_max = x.max(dim=dimension, keepdim=True).values

    # Subtract max before exponentiation to prevent overflow
    exp_x = torch.exp(x - x_max)

    # Sum along dimension
    return exp_x / exp_x.sum(-1, keepdim=True)
