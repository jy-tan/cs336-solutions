"""
Benchmark BPE tokenizer variants across different corpus sizes and vocab sizes.
Run: uv run python -m cs336_basics.benchmark_bpe
"""

import json
import os
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

from cs336_basics.tokenization import (
    train_bpe_tokenizer_1,
    train_bpe_tokenizer_2,
    train_bpe_tokenizer_3,
    train_bpe_tokenizer_4,
    train_bpe_tokenizer_5,
)


@dataclass
class BenchmarkResult:
    variant: str
    corpus_size_mb: float
    vocab_size: int
    time_seconds: float
    peak_memory_mb: float
    num_merges: int


def create_corpus_subset(input_path: str, output_path: str, size_mb: float) -> str:
    """Create a subset of the corpus with approximately the given size."""
    target_bytes = int(size_mb * 1024 * 1024)

    with open(input_path, encoding="utf-8") as f_in:
        content = f_in.read(target_bytes)

    # Find last complete document (ends with special token)
    last_doc_end = content.rfind("<|endoftext|>")
    if last_doc_end != -1:
        content = content[: last_doc_end + len("<|endoftext|>")]

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(content)

    actual_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Created subset: {actual_size:.2f} MB")
    return output_path


def benchmark_variant(
    variant_name: str,
    train_fn,
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> BenchmarkResult:
    """Benchmark a single BPE variant."""
    corpus_size_mb = os.path.getsize(input_path) / (1024 * 1024)

    tracemalloc.start()
    start_time = time.perf_counter()

    vocab, merges = train_fn(input_path, vocab_size, special_tokens, **kwargs)

    end_time = time.perf_counter()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        variant=variant_name,
        corpus_size_mb=corpus_size_mb,
        vocab_size=vocab_size,
        time_seconds=end_time - start_time,
        peak_memory_mb=peak_memory / (1024 * 1024),
        num_merges=len(merges),
    )


def get_corpus_stats(input_path: str, special_tokens: list[str]) -> dict:
    """Calculate W, L, P for a corpus (run once per corpus size)."""
    from cs336_basics.tokenization import pretokenize_chunk

    with open(input_path, encoding="utf-8") as f:
        corpus = f.read()

    word_frequencies = pretokenize_chunk(corpus, special_tokens)

    W = len(word_frequencies)
    total_tokens = sum(len(word) for word in word_frequencies.keys())
    L = total_tokens / W if W > 0 else 0

    # Count unique pairs
    pairs = set()
    for word in word_frequencies.keys():
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
    P = len(pairs)

    return {
        "corpus_path": input_path,
        "W": W,
        "L": round(L, 2),
        "P": P,
        "total_tokens": total_tokens,
    }


def run_benchmarks():
    # Configuration
    FULL_CORPUS = "./data/TinyStoriesV2-GPT4-valid.txt"  # Use valid for faster iteration
    SPECIAL_TOKENS = ["<|endoftext|>"]
    TEMP_DIR = Path("./bpe_benchmark")
    TEMP_DIR.mkdir(exist_ok=True)

    # Test matrix
    corpus_sizes_mb = [1, 5, 10, 25]  # Start small for naive
    vocab_sizes = [1000, 2000, 5000]

    # Variants to benchmark
    variants = {
        "v1": (train_bpe_tokenizer_1, {}),
        "v2": (train_bpe_tokenizer_2, {}),
        "v3": (train_bpe_tokenizer_3, {"num_processes": 4}),
        "v4": (train_bpe_tokenizer_4, {"num_processes": 4}),
        "v5": (train_bpe_tokenizer_5, {"num_processes": 4}),
    }

    results: list[BenchmarkResult] = []

    for corpus_size_mb in corpus_sizes_mb:
        # Create corpus subset
        subset_path = TEMP_DIR / f"corpus_{corpus_size_mb}mb.txt"
        create_corpus_subset(FULL_CORPUS, str(subset_path), corpus_size_mb)

        for vocab_size in vocab_sizes:
            for variant_name, (train_fn, kwargs) in variants.items():
                # Skip naive for large corpora (too slow)
                if variant_name == "naive" and corpus_size_mb > 10:
                    print(f"Skipping {variant_name} for {corpus_size_mb}MB (too slow)")
                    continue

                print(f"\n{'=' * 60}")
                print(f"Benchmarking: {variant_name}")
                print(f"  Corpus: {corpus_size_mb} MB, Vocab: {vocab_size}")
                print(f"{'=' * 60}")

                try:
                    result = benchmark_variant(
                        variant_name=variant_name,
                        train_fn=train_fn,
                        input_path=str(subset_path),
                        vocab_size=vocab_size,
                        special_tokens=SPECIAL_TOKENS,
                        **kwargs,
                    )
                    results.append(result)

                    print(f"  Time: {result.time_seconds:.2f}s")
                    print(f"  Peak Memory: {result.peak_memory_mb:.1f} MB")
                    print(f"  Merges: {result.num_merges}")

                except Exception as e:
                    print(f"  FAILED: {e}")

    results_path = TEMP_DIR / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(
            [
                {
                    "variant": r.variant,
                    "corpus_size_mb": r.corpus_size_mb,
                    "vocab_size": r.vocab_size,
                    "time_seconds": r.time_seconds,
                    "peak_memory_mb": r.peak_memory_mb,
                    "num_merges": r.num_merges,
                }
                for r in results
            ],
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")

    # Log corpus stats for each configuration
    corpus_stats = []
    for corpus_size_mb in corpus_sizes_mb:
        subset_path = TEMP_DIR / f"corpus_{corpus_size_mb}mb.txt"
        stats = get_corpus_stats(str(subset_path), SPECIAL_TOKENS)
        stats["corpus_size_mb"] = corpus_size_mb
        corpus_stats.append(stats)

    with open(TEMP_DIR / "corpus_stats.json", "w") as f:
        json.dump(corpus_stats, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Variant':<15} {'Corpus (MB)':<12} {'Vocab':<8} {'Time (s)':<12} {'Memory (MB)':<12}")
    print("-" * 80)
    for r in results:
        print(
            f"{r.variant:<15} {r.corpus_size_mb:<12.1f} {r.vocab_size:<8} {r.time_seconds:<12.2f} {r.peak_memory_mb:<12.1f}"
        )


if __name__ == "__main__":
    run_benchmarks()
