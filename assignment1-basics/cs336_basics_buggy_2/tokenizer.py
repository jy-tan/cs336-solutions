import pickle
import re
from collections.abc import Iterable, Iterator

import regex

PRE_TOKEN_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a BPE tokenizer from a given vocabulary, list of merges, and special tokens.

        Args:
            vocab: dict[int, bytes] - mapping from token ID to bytes
            merges: list[tuple[bytes, bytes]] - ordered list of BPE merges
            special_tokens: list[str] | None - list of special tokens to add/use
        """
        self.vocab = vocab.copy()
        self.merges = merges.copy()

        # Build reverse vocabulary for encoding (bytes -> token_id)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # Handle special tokens
        self.special_tokens = special_tokens if special_tokens else []

        # Add special tokens to vocab if not already present
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in self.inverse_vocab:
                new_id = len(self.vocab)
                self.vocab[new_id] = special_token_bytes
                self.inverse_vocab[special_token_bytes] = new_id

        # Build merge priority map for efficient encoding
        # Maps (token1, token2) -> (priority, merged_token_id)
        self.merge_priority = {}
        for priority, (token1, token2) in enumerate(self.merges):
            merged = token1 + token2
            if merged in self.inverse_vocab:
                token1_id = self.inverse_vocab[token1]
                token2_id = self.inverse_vocab[token2]
                merged_id = self.inverse_vocab[merged]
                self.merge_priority[(token1_id, token2_id)] = (priority, merged_id)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Class method that constructs and returns a Tokenizer from serialized files.

        Args:
            vocab_filepath: str - path to pickled vocabulary file
            merges_filepath: str - path to pickled merges file
            special_tokens: list[str] | None - list of special tokens

        Returns:
            Tokenizer instance
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, token_ids: list[int]) -> list[int]:
        """
        Apply BPE merges to a sequence of token IDs.

        Args:
            token_ids: list[int] - initial token IDs (usually byte-level)

        Returns:
            list[int] - token IDs after applying merges
        """
        if len(token_ids) <= 1:
            return token_ids

        while True:
            # Find the highest priority merge that can be applied
            best_pair = None
            best_priority = float("inf")
            best_pos = -1

            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self.merge_priority:
                    priority, _ = self.merge_priority[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = pair
                        best_pos = i

            # If no merge can be applied, we're done
            if best_pair is None:
                break

            # Apply the merge
            _, merged_id = self.merge_priority[best_pair]
            token_ids = token_ids[:best_pos] + [merged_id] + token_ids[best_pos + 2 :]

        return token_ids

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.

        Args:
            text: str - input text to encode

        Returns:
            list[int] - sequence of token IDs
        """
        if not text:
            return []

        token_ids = []

        # Split on special tokens first
        if self.special_tokens:
            pattern = (
                r"(" + "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)) + r")"
            )
            text_split = re.split(pattern, text)
        else:
            text_split = [text]

        for segment in text_split:
            if not segment:
                continue

            # Check if this segment is a special token
            if segment in self.special_tokens:
                special_token_bytes = segment.encode("utf-8")
                token_ids.append(self.inverse_vocab[special_token_bytes])
            else:
                # Pre-tokenize using the regex pattern
                for match in regex.finditer(PRE_TOKEN_REGEX, segment):
                    pre_token = match.group()
                    pre_token_bytes = pre_token.encode("utf-8")

                    # Convert to byte-level token IDs
                    byte_token_ids = [self.inverse_vocab[bytes([b])] for b in pre_token_bytes]

                    # Apply BPE merges
                    merged_ids = self._apply_merges(byte_token_ids)
                    token_ids.extend(merged_ids)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.
        This is memory-efficient for large files.

        Args:
            iterable: Iterable[str] - iterable of text strings

        Yields:
            int - token IDs one at a time
        """
        buffer = ""

        for chunk in iterable:
            buffer += chunk

            # Process complete pre-tokens from the buffer
            # We need to be careful not to split in the middle of a pre-token or special token

            # Find the last occurrence of a boundary we can safely split on
            # We'll look for the last whitespace or special token
            safe_split_pos = -1

            # Check for special tokens
            if self.special_tokens:
                for special_token in self.special_tokens:
                    pos = buffer.rfind(special_token)
                    if pos > safe_split_pos:
                        safe_split_pos = pos + len(special_token)

            # Look for whitespace boundaries (safer to split there)
            for i in range(len(buffer) - 1, -1, -1):
                if buffer[i].isspace():
                    safe_split_pos = max(safe_split_pos, i + 1)
                    break

            # If we found a safe split position and have enough buffer, process it
            if safe_split_pos > 0 and len(buffer) > 1000:  # arbitrary threshold
                text_to_process = buffer[:safe_split_pos]
                buffer = buffer[safe_split_pos:]

                for token_id in self.encode(text_to_process):
                    yield token_id

        # Process remaining buffer
        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.

        Args:
            ids: list[int] - sequence of token IDs

        Returns:
            str - decoded text
        """
        # Convert token IDs to bytes
        byte_sequence = b"".join(self.vocab[token_id] for token_id in ids)

        # Decode bytes to string
        return byte_sequence.decode("utf-8", errors="replace")
