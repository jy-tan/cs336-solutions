import argparse
import os

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import BPETokenizer


def tokenize_file(tokenizer: BPETokenizer, input_path: str, output_path: str, chunk_size: int = 1_000_000):
    """Tokenize a file in chunks with progress reporting and boundary-safe splitting."""

    file_size = os.path.getsize(input_path)
    print(f"Tokenizing {input_path} ({file_size / 1e6:.1f} MB)...")

    all_tokens = []
    buffer = ""
    bytes_processed = 0

    with open(input_path, encoding="utf-8") as f:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Tokenizing") as pbar:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                buffer += chunk

                # Find the last newline to split safely
                last_newline = buffer.rfind("\n")

                if last_newline != -1:
                    # Process everything up to and including the last newline
                    to_process = buffer[: last_newline + 1]
                    buffer = buffer[last_newline + 1 :]

                    tokens = tokenizer.encode(to_process)
                    all_tokens.extend(tokens)

                chunk_bytes = len(chunk.encode("utf-8"))
                bytes_processed += chunk_bytes
                pbar.update(chunk_bytes)

    # Process any remaining buffer
    if buffer:
        tokens = tokenizer.encode(buffer)
        all_tokens.extend(tokens)

    max_id = max(all_tokens) if all_tokens else 0
    dtype = np.uint16 if max_id < 65536 else np.uint32

    tokens_array = np.array(all_tokens, dtype=dtype)

    print(f"  {len(tokens_array):,} tokens")
    print(f"  Dtype: {dtype}")

    np.save(output_path, tokens_array)
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Tokenize text files for training")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab .pkl file")
    parser.add_argument("--merges", type=str, required=True, help="Path to merges .pkl file")
    parser.add_argument("--input", type=str, required=True, help="Input .txt file")
    parser.add_argument("--output", type=str, required=True, help="Output .npy file")
    parser.add_argument(
        "--special-tokens", type=str, nargs="*", default=["<|endoftext|>"], help="Special tokens to use"
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Chunk size in characters")

    args = parser.parse_args()

    tokenizer = BPETokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=args.special_tokens,
    )

    print(f"Loaded tokenizer with {len(tokenizer.vocab)} tokens")

    tokenize_file(tokenizer, args.input, args.output, args.chunk_size)


if __name__ == "__main__":
    main()
