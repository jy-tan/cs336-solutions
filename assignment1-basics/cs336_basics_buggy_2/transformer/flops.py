def calculate_transformer_flops(
    batch_size: int, seq_len: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, vocab_size: int
):
    """
    Calculate the FLOPs for a Transformer model.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        num_layers: Number of transformer blocks
        vocab_size: Vocabulary size

    Returns:
        dict: Dictionary mapping component names to their FLOPs
    """
    # Per-block calculations
    layer_norm = 4 * batch_size * seq_len * d_model

    # Attention breakdown
    attention_projections = 8 * batch_size * seq_len * d_model**2
    attention_rope = 6 * batch_size * seq_len * d_model
    attention_mechanism = 4 * batch_size * seq_len**2 * d_model
    attention_softmax = 6 * batch_size * num_heads * seq_len**2

    # Feed-forward
    feedforward = 6 * batch_size * seq_len * d_model * d_ff

    # Residual additions
    residual_adds = 2 * batch_size * seq_len * d_model

    per_block_total = (
        2 * layer_norm  # ln1 and ln2
        + attention_projections
        + attention_rope
        + attention_mechanism
        + attention_softmax
        + feedforward
        + residual_adds
    )

    final_norm = 4 * batch_size * seq_len * d_model
    lm_head = 2 * batch_size * seq_len * d_model * vocab_size

    total_blocks_flops = num_layers * per_block_total
    total_flops = total_blocks_flops + final_norm + lm_head

    counts = {
        "attention_projections": num_layers * attention_projections,
        "attention_mechanism": num_layers * attention_mechanism,
        "attention_softmax": num_layers * attention_softmax,
        "attention_rope": num_layers * attention_rope,
        "feedforward": num_layers * feedforward,
        "layer_norms": num_layers * 2 * layer_norm + final_norm,
        "residual_adds": num_layers * residual_adds,
        "lm_head": lm_head,
    }

    # Print breakdown
    print(f"Total FLOPs: {total_flops:,}")
    print("\nFLOPs breakdown by component:")
    print("=" * 60)

    # Sort by FLOPs (descending)
    sorted_components = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    for component, flops in sorted_components:
        percentage = (flops / total_flops) * 100
        print(f"{component:25s}: {flops:15,} ({percentage:5.2f}%)")

    print("=" * 60)
    print(f"{'TOTAL':25s}: {total_flops:15,} (100.00%)")

    return counts


if __name__ == "__main__":
    # Fix for all models
    batch_size = 1
    vocab_size = 50257
    # seq_len = 1024
    seq_len = 16384

    def calculate_d_ff(d_model: int, ratio: float = 8 / 3) -> int:
        """Calculate d_ff as `ratio * d_model`, rounded to nearest multiple of 64."""
        d_ff_approx = ratio * d_model
        return round(d_ff_approx / 64) * 64

    d_ff_ratio = 4

    # GPT-2 small
    gpt2_small_config = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": 768,
        "num_heads": 12,
        "d_ff": calculate_d_ff(768, d_ff_ratio),
        "num_layers": 12,
        "vocab_size": vocab_size,
    }

    # GPT-2 medium
    gpt2_medium_config = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": 1024,
        "num_heads": 12,
        "d_ff": calculate_d_ff(1024, d_ff_ratio),
        "num_layers": 24,
        "vocab_size": vocab_size,
    }

    # GPT-2 large
    gpt2_large_config = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": 1280,
        "num_heads": 20,
        "d_ff": calculate_d_ff(1280, d_ff_ratio),
        "num_layers": 36,
        "vocab_size": vocab_size,
    }

    # GPT-2 XL
    gpt2_xl_config = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": 1600,
        "num_heads": 25,
        "d_ff": calculate_d_ff(1600, d_ff_ratio),
        "num_layers": 48,
        "vocab_size": vocab_size,
    }

    print("GPT-2 small:")
    counts = calculate_transformer_flops(**gpt2_small_config)
    print("=" * 60, "\n")

    print("GPT-2 medium:")
    counts = calculate_transformer_flops(**gpt2_medium_config)
    print("=" * 60, "\n")

    print("GPT-2 large:")
    counts = calculate_transformer_flops(**gpt2_large_config)
    print("=" * 60, "\n")

    print("GPT-2 XL:")
    counts = calculate_transformer_flops(**gpt2_xl_config)
    print("=" * 60, "\n")
