import torch
from einops import rearrange, reduce
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy_loss(logits: Float[Tensor, " ... seq_len vocab_size"], targets: Int[Tensor, " ... seq_len"]):
    """
    Compute the cross-entropy loss for a sequence of logits and targets.

    Args:
        logits: Float[Tensor, " ... batch_size vocab_size"], the logits output by the model.
        targets: Int[Tensor, " batch_size"], the targets indices.

    Returns:
        Float[Tensor, ""], the cross-entropy loss.
    """
    logits_flat: Float[Tensor, "batch_seq_flat vocab_size"] = rearrange(logits, "... vocab_size -> (...) vocab_size")
    targets_flat: Int[Tensor, " batch_seq_flat"] = rearrange(targets, "... -> (...)")

    # Subtract max for numerical stability
    max_logits: Float[Tensor, " batch_seq_flat 1"] = reduce(
        logits_flat, " batch_seq_flat vocab_size -> batch_seq_flat 1", "max"
    )
    logits_shifted = logits_flat - max_logits
    log_sum_exp: Float[Tensor, " batch_seq_flat 1"] = torch.log(torch.exp(logits_shifted).sum(dim=-1, keepdim=True))

    # Log softmax
    log_probs: Float[Tensor, " batch_seq_flat vocab_size"] = logits_shifted - log_sum_exp

    batch_seq_size = logits_flat.shape[0]
    target_log_probs: Float[Tensor, " batch_seq_flat"] = log_probs[torch.arange(batch_seq_size), targets_flat]

    loss = target_log_probs.mean()  # BUG: should be negative here

    return loss
