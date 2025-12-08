"""
Plot V4 vs V5 crossover point with fine-grained vocab sizes.
Run: uv run python -m cs336_basics.bpe_heap_crossover
"""

import json
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cs336_basics.tokenization import (
    train_bpe_tokenizer_4,
    train_bpe_tokenizer_5,
)


def benchmark_variant(train_fn, input_path: str, vocab_size: int, special_tokens: list[str], **kwargs) -> float:
    """Benchmark and return time in seconds."""
    tracemalloc.start()
    start_time = time.perf_counter()
    train_fn(input_path, vocab_size, special_tokens, **kwargs)
    end_time = time.perf_counter()
    tracemalloc.stop()
    return end_time - start_time


def create_corpus_subset(input_path: str, output_path: str, size_mb: float) -> str:
    """Create a subset of the corpus with approximately the given size."""
    target_bytes = int(size_mb * 1024 * 1024)

    with open(input_path, encoding="utf-8") as f_in:
        content = f_in.read(target_bytes)

    last_doc_end = content.rfind("<|endoftext|>")
    if last_doc_end != -1:
        content = content[: last_doc_end + len("<|endoftext|>")]

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(content)

    return output_path


def run_crossover_benchmark():
    FULL_CORPUS = "./data/TinyStoriesV2-GPT4-valid.txt"
    SPECIAL_TOKENS = ["<|endoftext|>"]
    TEMP_DIR = Path("./bpe_benchmark")
    TEMP_DIR.mkdir(exist_ok=True)

    vocab_sizes = [500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000, 5000]
    corpus_sizes_mb = [5, 10, 21]  # Skip 1MB since V5 wins everywhere there

    results = {}

    for corpus_mb in corpus_sizes_mb:
        subset_path = TEMP_DIR / f"corpus_{corpus_mb}mb.txt"
        if not subset_path.exists():
            create_corpus_subset(FULL_CORPUS, str(subset_path), corpus_mb)

        results[corpus_mb] = {"v4": [], "v5": [], "vocab_sizes": []}

        for vocab_size in vocab_sizes:
            print(f"Benchmarking corpus={corpus_mb}MB, vocab={vocab_size}...")

            t4 = benchmark_variant(train_bpe_tokenizer_4, str(subset_path), vocab_size, SPECIAL_TOKENS, num_processes=4)
            t5 = benchmark_variant(train_bpe_tokenizer_5, str(subset_path), vocab_size, SPECIAL_TOKENS, num_processes=4)

            results[corpus_mb]["v4"].append(t4)
            results[corpus_mb]["v5"].append(t5)
            results[corpus_mb]["vocab_sizes"].append(vocab_size)

            print(f"  V4: {t4:.2f}s, V5: {t5:.2f}s, Winner: {'V5' if t5 < t4 else 'V4'}")

    with open(TEMP_DIR / "crossover_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def find_crossover(vocab_sizes, v4_times, v5_times):
    """Find crossover point via linear interpolation."""
    diff = np.array(v4_times) - np.array(v5_times)

    for i in range(len(diff) - 1):
        if diff[i] < 0 and diff[i + 1] >= 0:
            # V4 was winning, now V5 is winning (shouldn't happen based on data)
            t = -diff[i] / (diff[i + 1] - diff[i])
            return vocab_sizes[i] + t * (vocab_sizes[i + 1] - vocab_sizes[i])
        elif diff[i] >= 0 and diff[i + 1] < 0:
            # V5 was winning, now V4 is winning (also shouldn't happen)
            t = diff[i] / (diff[i] - diff[i + 1])
            return vocab_sizes[i] + t * (vocab_sizes[i + 1] - vocab_sizes[i])
        elif diff[i] <= 0 and diff[i + 1] > 0:
            # V4 was faster or tied, now V5 is faster
            t = -diff[i] / (diff[i + 1] - diff[i])
            return vocab_sizes[i] + t * (vocab_sizes[i + 1] - vocab_sizes[i])

    return None


def plot_crossover(results: dict, output_path: str = "benchmark_temp/crossover_plot.png"):
    """Create publication-quality crossover plot."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5), sharey=False)

    if len(results) == 1:
        axes = [axes]

    colors = {"v4": "#e74c3c", "v5": "#3498db"}

    crossover_points = []

    for idx, (corpus_mb, data) in enumerate(sorted(results.items())):
        ax = axes[idx]
        vocab_sizes = data["vocab_sizes"]
        v4_times = data["v4"]
        v5_times = data["v5"]

        # Plot lines
        ax.plot(vocab_sizes, v4_times, "o-", color=colors["v4"], label="V4 (Inverted Index)", linewidth=2, markersize=6)
        ax.plot(vocab_sizes, v5_times, "s-", color=colors["v5"], label="V5 (Heap)", linewidth=2, markersize=6)

        # Find and mark crossover
        crossover = find_crossover(vocab_sizes, v4_times, v5_times)
        if crossover:
            crossover_points.append((corpus_mb, crossover))
            # Interpolate y value at crossover
            y_cross = np.interp(crossover, vocab_sizes, v4_times)
            ax.axvline(x=crossover, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
            ax.scatter([crossover], [y_cross], color="black", s=100, zorder=5, marker="x")
            ax.annotate(
                f"Crossover\n≈{crossover:.0f}",
                xy=(crossover, y_cross),
                xytext=(15, 15),
                textcoords="offset points",
                fontsize=10,
                ha="left",
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
            )

        # Shade regions
        ax.fill_between(
            vocab_sizes,
            v4_times,
            v5_times,
            where=[v4 > v5 for v4, v5 in zip(v4_times, v5_times)],
            alpha=0.15,
            color=colors["v5"],
            label="_V5 wins",
        )
        ax.fill_between(
            vocab_sizes,
            v4_times,
            v5_times,
            where=[v4 <= v5 for v4, v5 in zip(v4_times, v5_times)],
            alpha=0.15,
            color=colors["v4"],
            label="_V4 wins",
        )

        ax.set_xlabel("Vocabulary Size", fontsize=12)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_title(f"Corpus: {corpus_mb} MB", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min(vocab_sizes) - 100, max(vocab_sizes) + 100)

    plt.suptitle("V4 vs V5 Performance Crossover", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("CROSSOVER SUMMARY")
    print("=" * 50)
    for corpus_mb, crossover in crossover_points:
        print(f"  {corpus_mb} MB corpus: V5 beats V4 when vocab > ~{crossover:.0f}")

    if crossover_points:
        avg_crossover = sum(c for _, c in crossover_points) / len(crossover_points)
        print(f"\n  Average crossover point: ~{avg_crossover:.0f}")

    return crossover_points


def main():
    TEMP_DIR = Path("./bpe_benchmark")
    results_path = TEMP_DIR / "crossover_results.json"

    # Check if we have cached results
    if results_path.exists():
        print("Loading cached crossover results...")
        with open(results_path) as f:
            results = json.load(f)
        # Convert string keys back to int
        results = {int(k): v for k, v in results.items()}
    else:
        print("Running crossover benchmarks (this may take a while)...")
        results = run_crossover_benchmark()

    plot_crossover(results)


if __name__ == "__main__":
    main()
