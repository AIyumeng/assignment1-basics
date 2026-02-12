import os
import regex as re
from collections import Counter
from typing import Dict, List, Tuple


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # ---------- 1. init vocab ----------
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    vocab_rev: Dict[bytes, int] = {v: k for k, v in vocab.items()}

    # add special tokens
    for tok in special_tokens:
        b = tok.encode("utf-8")
        idx = len(vocab)
        vocab[idx] = b
        vocab_rev[b] = idx

    num_merges = vocab_size - len(vocab)
    merges: List[Tuple[bytes, bytes]] = []

    # ---------- 2. load data ----------
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # split by special tokens (do not lose them)
    if special_tokens:
        pattern = "|".join(re.escape(t) for t in special_tokens)
        chunks = re.split(f"({pattern})", text)
    else:
        chunks = [text]

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    word_counter = Counter()
    word_encode: Dict[str, List[int]] = {}

    for chunk in chunks:
        if chunk in special_tokens:
            word_counter[chunk] += 1
            word_encode[chunk] = [vocab_rev[chunk.encode("utf-8")]]
        else:
            for m in re.finditer(PAT, chunk):
                w = m.group(0)
                word_counter[w] += 1
                word_encode[w] = list(w.encode("utf-8"))

    # ---------- 3. helper: merge ----------
    def merge_word(tokens: List[int], pair: Tuple[int, int], new_id: int):
        out = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    # ---------- 4. BPE loop ----------
    for _ in range(num_merges):
        pair_counter = Counter()

        for w, freq in word_counter.items():
            tokens = word_encode[w]
            for a, b in zip(tokens[:-1], tokens[1:]):
                pair_counter[(a, b)] += freq

        if not pair_counter:
            break

        best_pair = max(
            pair_counter, key=lambda p: (pair_counter[p], vocab[p[0]], vocab[p[1]])
        )

        new_id = len(vocab)
        new_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[new_id] = new_bytes
        vocab_rev[new_bytes] = new_id

        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        for w in word_encode:
            word_encode[w] = merge_word(word_encode[w], best_pair, new_id)

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe(
        "/Users/dym/Desktop/Project/assignment1-basics/data/test.txt",
        280,
        ["<|endoftext|>"],
    )

    print("Total merges:", len(merges))
    print(merges)
