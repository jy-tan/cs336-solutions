import json
import os
import re
from multiprocessing import Pool

import regex
from tqdm import tqdm

PRE_TOKEN_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe_tokenizer_naive(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> (dict[int, bytes], list[tuple[bytes, bytes]]):
    """
    Train a BPE tokenizer on the input data.

    Args:
        input_path (str | os.PathLike): The path to the input data.
        vocab_size (int): The size of the vocabulary.
        special_tokens (list[str]): The special tokens to use.

    Returns:
        vocab (dict[int, bytes]): The tokenizer vocabulary.
        merges (list[tuple[bytes, bytes]]): List of BPE merges produced from training, ordered by order of creation.
    """
    # Read the file
    with open(input_path, encoding="utf-8") as f:
        corpus = f.read()

    # Split corpus on the special tokens
    if special_tokens:
        pattern = r"(" + "|".join(re.escape(token) for token in special_tokens) + r")"
        corpus_split: list[str] = re.split(pattern, corpus)
    else:
        corpus_split = [corpus]
    corpus_split = [token for token in corpus_split if token]

    vocab = {idx: bytes([idx]) for idx in range(256)}

    # Note: keys are tuples of ints, each int is a token id.
    # This can exceed 255 after merges
    word_frequencies: dict[tuple[int, ...], int] = {}

    for document in corpus_split:
        if document in special_tokens:
            continue

        for match in regex.finditer(PRE_TOKEN_REGEX, document):
            token = match.group()
            token_bytes = tuple[int, ...](token.encode("utf-8"))

            word_frequencies[token_bytes] = word_frequencies.get(token_bytes, 0) + 1

    # Add special tokens to the vocabulary
    special_token_id = len(vocab)
    for special_token in special_tokens:
        vocab[special_token_id] = special_token.encode("utf-8")
        special_token_id += 1

    merges: list[tuple[bytes, bytes]] = []

    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        # Compute frequency of adjacent pairs
        pair_frequencies: dict[tuple[int, int], int] = {}
        for word, frequency in word_frequencies.items():
            # word is a tuple of ints representing a word split by regex
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])

                # If a pair belongs to a word that occurs `frequency` times, add it to the pair frequency as well.
                pair_frequencies[pair] = pair_frequencies.get(pair, 0) + frequency

        # No pairs found from all words, so we're done.
        if not pair_frequencies:
            break

        # Decide on which pair to merge
        # This is the one with the highest frequency, break ties lexicographically.
        highest_frequency_pair = max(
            pair_frequencies.keys(), key=lambda pair: (pair_frequencies[pair], vocab[pair[0]], vocab[pair[1]])
        )

        # Add new token id for this merge, then add it to the vocabulary
        new_id = len(vocab)
        vocab[new_id] = vocab[highest_frequency_pair[0]] + vocab[highest_frequency_pair[1]]

        # Add to merges list
        merges.append((vocab[highest_frequency_pair[0]], vocab[highest_frequency_pair[1]]))

        # Merge all occurrences of the pair and create new word frequency dict
        # We're basically just transforming the keys of the dictionary here, values stay the same.
        new_word_frequencies: dict[tuple[int, ...], int] = {}
        for word, frequency in word_frequencies.items():
            new_word: list[int] = []
            i = 0
            while i < len(word):
                # Check if we can merge at this position
                if i < len(word) - 1 and (word[i], word[i + 1]) == highest_frequency_pair:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word_frequencies[tuple[int, ...](new_word)] = frequency

        word_frequencies = new_word_frequencies

    return vocab, merges


def train_bpe_tokenizer_optimized(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input data.
    The merging step is optimized to only compute the frequency of pairs that are affected by each merge.

    Args:
        input_path (str | os.PathLike): The path to the input data.
        vocab_size (int): The size of the vocabulary.
        special_tokens (list[str]): The special tokens to use.

    Returns:
        vocab (dict[int, bytes]): The tokenizer vocabulary.
        merges (list[tuple[bytes, bytes]]): List of BPE merges produced from training, ordered by order of creation.
    """
    with open(input_path, encoding="utf-8") as f:
        corpus = f.read()

    if special_tokens:
        pattern = r"(" + "|".join(re.escape(token) for token in special_tokens) + r")"
        corpus_split: list[str] = re.split(pattern, corpus)
    else:
        corpus_split = [corpus]
    corpus_split = [token for token in corpus_split if token]

    vocab = {idx: bytes([idx]) for idx in range(256)}

    word_frequencies: dict[tuple[int, ...], int] = {}

    for document in corpus_split:
        if document in special_tokens:
            continue

        for match in regex.finditer(PRE_TOKEN_REGEX, document):
            token = match.group()
            token_bytes = tuple[int, ...](token.encode("utf-8"))

            word_frequencies[token_bytes] = word_frequencies.get(token_bytes, 0) + 1

    special_token_id = len(vocab)
    for special_token in special_tokens:
        vocab[special_token_id] = special_token.encode("utf-8")
        special_token_id += 1

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    # Above is the same as the naive implementatio.

    # Build pair frequency cache once, then reuse it to update incrementally.
    pair_frequencies: dict[tuple[int, int], int] = {}

    # Helper function get all pairs of adjacent tokens in a word
    def get_pairs(word: tuple[int, ...]) -> list[tuple[int, int]]:
        return [(word[i], word[i + 1]) for i in range(len(word) - 1)]

    for word, frequency in word_frequencies.items():
        for pair in get_pairs(word):
            pair_frequencies[pair] = pair_frequencies.get(pair, 0) + frequency

    def merge_word(word: tuple[int, ...], pair: tuple[int, int]) -> tuple[int, ...]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple[int, ...](new_word)

    for _ in range(num_merges):
        if not pair_frequencies:
            break

        highest_frequency_pair = max(
            pair_frequencies.keys(), key=lambda pair: (pair_frequencies[pair], vocab[pair[0]], vocab[pair[1]])
        )

        new_id = len(vocab)
        vocab[new_id] = vocab[highest_frequency_pair[0]] + vocab[highest_frequency_pair[1]]
        merges.append((vocab[highest_frequency_pair[0]], vocab[highest_frequency_pair[1]]))

        # Update pair frequencies. Only process words that contain the pair to merge.
        new_word_frequencies: dict[tuple[int, ...], int] = {}

        for word, frequency in word_frequencies.items():
            if highest_frequency_pair not in get_pairs(word):
                # Word is unchanged, we just copy it over
                new_word_frequencies[word] = frequency

            else:
                # Subtract old pair counts from pair frequency
                for pair in get_pairs(word):
                    pair_frequencies[pair] -= frequency

                    if pair_frequencies[pair] == 0:
                        del pair_frequencies[pair]

                new_word = merge_word(word, highest_frequency_pair)
                new_word_frequencies[new_word] = frequency

                # Add the new pair counts to pair frequency
                for pair in get_pairs(new_word):
                    pair_frequencies[pair] = pair_frequencies.get(pair, 0) + frequency

        word_frequencies = new_word_frequencies

    return vocab, merges


def pretokenize_chunk(chunk: str, special_tokens: list[str]) -> dict[tuple[int, ...], int]:
    """
    Pre-tokenize a chunk of text and return word frequencies.
    This function will be run in parallel by multiple processes.
    """
    word_frequencies: dict[tuple[int, ...], int] = {}

    # Split chunk on special tokens
    if special_tokens:
        pattern = r"(" + "|".join(re.escape(token) for token in special_tokens) + r")"
        corpus_split = re.split(pattern, chunk)
    else:
        corpus_split = [chunk]
    corpus_split = [token for token in corpus_split if token]

    for document in corpus_split:
        if document in special_tokens:
            continue

        for match in regex.finditer(PRE_TOKEN_REGEX, document):
            token = match.group()
            token_bytes = tuple[int, ...](token.encode("utf-8"))
            word_frequencies[token_bytes] = word_frequencies.get(token_bytes, 0) + 1

    return word_frequencies


def _pretokenize_chunk_wrapper(args):
    """Wrapper to unpack arguments for multiprocessing.imap."""
    chunk, special_tokens = args
    return pretokenize_chunk(chunk, special_tokens)


def find_chunk_boundaries(file, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """Find chunk boundaries at special token locations."""
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set[int](chunk_boundaries))


def save_token_frequencies(word_frequencies: dict[tuple[int, ...], int], vocab: dict[int, bytes], save_path: str):
    """
    Save the frequency of each token ID in the vocabulary.

    Args:
        word_frequencies: Dictionary mapping word tuples (sequences of token IDs) to their frequencies
        vocab: The tokenizer vocabulary mapping token IDs to bytes
        save_path: Path to save the token frequencies
    """
    token_frequencies: dict[int, int] = {}

    for word_tuple, word_freq in word_frequencies.items():
        # For each token ID in this word
        for token_id in word_tuple:
            # Add the word's frequency to this token's count
            token_frequencies[token_id] = token_frequencies.get(token_id, 0) + word_freq

    # Convert to list for all tokens in vocab (including zero-frequency ones)
    token_freq_list = []
    for token_id, token_bytes in vocab.items():
        freq = token_frequencies.get(token_id, 0)
        token_str = token_bytes.decode("utf-8", errors="replace")
        token_freq_list.append({"token_id": token_id, "token": token_str, "frequency": freq})

    token_freq_list.sort(key=lambda x: x["frequency"], reverse=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for entry in token_freq_list:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"\nSaved {len(token_freq_list)} token frequencies to {save_path}")


def train_bpe_tokenizer_parallel(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
    save_frequencies_path: str | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input data. Uses parallel pre-tokenization.

    Args:
        input_path (str | os.PathLike): The path to the input data.
        vocab_size (int): The size of the vocabulary.
        special_tokens (list[str]): The special tokens to use.
        num_processes (int | None): The number of processes to use. If None, use the number of CPU cores.
        save_frequencies_path (str | None): The path to save the vocabulary token frequencies. If None, do not save.

    Returns:
        vocab (dict[int, bytes]): Tokenizer vocabuluary produced from training.
        merges (list[tuple[bytes, bytes]]): List of BPE merges produced from training, ordered by order of creation.
    """
    if save_frequencies_path is not None:
        if not os.path.exists(os.path.dirname(save_frequencies_path)):
            os.makedirs(os.path.dirname(save_frequencies_path), exist_ok=True)
        if not save_frequencies_path.endswith(".jsonl"):
            save_frequencies_path += ".jsonl"

    if num_processes is None:
        num_processes = os.cpu_count() or 4

    num_chunks = num_processes * 3

    chunks: list[str] = []
    chunk_frequencies: list[dict[tuple[int, ...], int]] = []

    with open(input_path, "rb") as f:
        boundary_token = special_tokens[0].encode("utf-8")
        chunk_boundaries = find_chunk_boundaries(f, num_chunks, boundary_token)

        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            f.seek(start)
            chunk: str = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    # with Pool(num_processes) as pool:
    #     chunk_frequencies = pool.starmap(pretokenize_chunk, [(chunk, special_tokens) for chunk in chunks])
    with Pool(num_processes) as pool:
        args = [(chunk, special_tokens) for chunk in chunks]

        chunk_frequencies = list(
            tqdm(
                pool.imap(_pretokenize_chunk_wrapper, args, chunksize=1), total=len(chunks), desc="Pretokenizing chunks"
            )
        )

    # Combine word frequencies from all chunks
    word_frequencies: dict[tuple[int, ...], int] = {}
    for chunk_frequency in chunk_frequencies:
        for word, frequency in chunk_frequency.items():
            word_frequencies[word] = word_frequencies.get(word, 0) + frequency

    vocab = {idx: bytes([idx]) for idx in range(256)}

    special_token_id = len(vocab)
    for special_token in special_tokens:
        vocab[special_token_id] = special_token.encode("utf-8")
        special_token_id += 1

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    # Below is the same as `train_bpe_tokenizer_optimized`

    pair_frequencies: dict[tuple[int, int], int] = {}

    def get_pairs(word: tuple[int, ...]) -> list[tuple[int, int]]:
        return [(word[i], word[i + 1]) for i in range(len(word) - 1)]

    for word, frequency in word_frequencies.items():
        for pair in get_pairs(word):
            pair_frequencies[pair] = pair_frequencies.get(pair, 0) + frequency

    def merge_word(word: tuple[int, ...], pair: tuple[int, int]) -> tuple[int, ...]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple[int, ...](new_word)

    for _ in tqdm(range(num_merges), desc="Computing merges", unit="merge"):
        if not pair_frequencies:
            break

        highest_frequency_pair = max(
            pair_frequencies.keys(), key=lambda pair: (pair_frequencies[pair], vocab[pair[0]], vocab[pair[1]])
        )

        new_id = len(vocab)
        vocab[new_id] = vocab[highest_frequency_pair[0]] + vocab[highest_frequency_pair[1]]
        merges.append((vocab[highest_frequency_pair[0]], vocab[highest_frequency_pair[1]]))

        new_word_frequencies: dict[tuple[int, ...], int] = {}

        for word, frequency in word_frequencies.items():
            if highest_frequency_pair not in get_pairs(word):
                new_word_frequencies[word] = frequency

            else:
                for pair in get_pairs(word):
                    pair_frequencies[pair] -= frequency

                    if pair_frequencies[pair] == 0:
                        del pair_frequencies[pair]

                new_word = merge_word(word, highest_frequency_pair)
                new_word_frequencies[new_word] = frequency

                # Add the new pair counts to pair frequency
                for pair in get_pairs(new_word):
                    pair_frequencies[pair] = pair_frequencies.get(pair, 0) + frequency

        word_frequencies = new_word_frequencies

    if save_frequencies_path:
        save_token_frequencies(word_frequencies, vocab, save_frequencies_path)

    return vocab, merges


if __name__ == "__main__":
    import os
    import pickle
    import time
    import tracemalloc

    # INPUT_PATH = "./data/TinyStoriesV2-GPT4-train.txt"
    INPUT_PATH = "./data/TinyStoriesV2-GPT4-valid.txt"
    # INPUT_PATH = "./data/owt_train.txt"
    VOCAB_SIZE = 10_000
    # VOCAB_SIZE = 32_000
    SPECIAL_TOKENS = ["<|endoftext|>"]

    def sanitize_filename(filename: str) -> str:
        # Remove directory, remove file extension, and replace non-alphanum with underscores
        base = os.path.basename(filename)
        base_no_ext = os.path.splitext(base)[0]
        return "".join([c if c.isalnum() else "_" for c in base_no_ext])

    RESULTS_DIR = "./results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_file_name = sanitize_filename(INPUT_PATH)

    num_processes = os.cpu_count()

    print("Training BPE tokenizer...")
    print(f"- input path: {INPUT_PATH}")
    print(f"- vocab size: {VOCAB_SIZE}")
    print(f"- special tokens: {SPECIAL_TOKENS}")
    print(f"- num processes: {num_processes}\n")

    tracemalloc.start()
    start_time = time.time()

    vocab, merges = train_bpe_tokenizer_parallel(
        INPUT_PATH,
        VOCAB_SIZE,
        SPECIAL_TOKENS,
        save_frequencies_path=f"{RESULTS_DIR}/{save_file_name}_token_frequencies.jsonl",
    )

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Current memory: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {longest_token} (length: {len(longest_token)} bytes)")
    print(f"Decoded (if valid UTF-8): {longest_token.decode('utf-8', errors='replace')}")

    with open(f"{RESULTS_DIR}/{save_file_name}_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(f"{RESULTS_DIR}/{save_file_name}_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
