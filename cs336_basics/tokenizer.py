import regex as re
from typing import List, Iterable, Iterator
import pickle

GPT2_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.vocab_rev = {v: k for k, v in vocab.items()}

        # special tokens：最长优先
        self._special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)

    def from_file(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    # ---------- special token split ----------
    def _split_by_special_tokens(self, text: str) -> List[str]:
        if not self.special_tokens:
            return [text]

        # 注意 escape + 最长优先
        pattern = "|".join(re.escape(tok) for tok in self._special_tokens_sorted)
        pieces = re.split(f"({pattern})", text)

        return pieces

    # ---------- BPE ----------
    def _bpe_encode_bytes(self, b: bytes) -> List[int]:
        tokens = [bytes([x]) for x in b]

        for a, b_ in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b_:
                    new_tokens.append(a + b_)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self.vocab_rev[t] for t in tokens]

    # ---------- encode ----------
    def encode(self, text: str) -> List[int]:
        ids: List[int] = []

        # ① special token lexing
        pieces = self._split_by_special_tokens(text)

        for piece in pieces:
            if piece in self.special_tokens:
                ids.append(self.vocab_rev[piece.encode("utf-8")])
                continue

            # ② GPT-2 regex 预分词
            for m in re.finditer(GPT2_PATTERN, piece):
                span = m.group(0)
                ids.extend(self._bpe_encode_bytes(span.encode("utf-8")))

        return ids

    # ---------- encode iterable ----------
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for x in iterable:
            yield from self.encode(x)

    # ---------- decode ----------
    def decode(self, ids: List[int]) -> str:
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")
