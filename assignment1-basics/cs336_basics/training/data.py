import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor


def get_batch(
    x: np.array, batch_size: int, context_length: int, device: torch.device | str
) -> (Float[Tensor, " batch_size context_length"], Float[Tensor, " batch_size context_length"]):
    """
    Data loader that samples random batches from a tokenized input sequence.

    Args:
        x: numpy array of token IDs
        batch_size: number of sequences to sample
        context_length: length of each sampled sequence
        device: PyTorch device to place the tensors on
    """
    max_start_index = len(x) - context_length

    start_indices = np.random.randint(0, max_start_index, size=batch_size)

    inputs = np.stack([x[i : i + context_length] for i in start_indices])
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)

    targets = np.stack([x[i + 1 : i + context_length + 1] for i in start_indices])
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets
