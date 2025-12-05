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

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Int[Tensor, "batch seq_len"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
    ) -> Int[Tensor, "batch new_seq_len"]:
        """
        Generate a sequence of tokens from the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature. Use 0 for greedy decoding.
            top_p: Nucleus sampling threshold (0 < p <= 1)
            eos_token_id: Stop generation when all sequences produce this token

        Returns:
            Token IDs including prompt and generated tokens
        """
        for _ in range(max_new_tokens):
            # Truncate to context length if needed (sliding window)
            context_length = self.layers[0].attn.max_seq_len  # or however you store it
            if input_ids.shape[1] > context_length:
                input_ids_cond = input_ids[:, -context_length:]
            else:
                input_ids_cond = input_ids

            logits: Float[Tensor, "batch seq_len vocab_size"] = self.forward(input_ids_cond)

            # Logits for the next last position, can we use einops here?
            next_token_logits: Float[Tensor, "batch vocab_size"] = logits[:, -1, :]

            if temperature == 0:  # Greedy
                next_tokens = next_token_logits.argmax(dim=-1)
            else:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                probs: Float[Tensor, "batch vocab_size"] = torch.softmax(next_token_logits, dim=-1)

                if top_p < 1.0:
                    next_tokens = self._sample_top_p(probs, top_p)
                else:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break

        return input_ids

    def _sample_top_p(self, probs: Float[Tensor, "batch vocab_size"], p: float):
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask for tokens to remove, True = mask
        # Shift right by 1 to include the token that crosses the threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Apply mask and normalize
        sorted_probs[sorted_indices_to_remove] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # Sample from filtered distribution
        sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)

        batch_indices = torch.arange(probs.shape[0], device=probs.device)
        next_tokens = sorted_indices[batch_indices, sampled_sorted_idx]

        return next_tokens
