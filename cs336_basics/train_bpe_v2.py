import os
import regex as re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, BinaryIO
from multiprocessing import Pool, cpu_count
from tqdm import trange
import time


# ===============================
# chunk 边界函数找到分隔符位置
# ===============================
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:

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

    return sorted(set(chunk_boundaries))


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# ===============================
# 预分词统计词频
# ===============================
def process_chunk(args):
    chunk, special_tokens = args

    word_counter = Counter()

    if special_tokens:
        pattern = "|".join(re.escape(t) for t in special_tokens)
        pieces = re.split(f"({pattern})", chunk)
    else:
        pieces = [chunk]

    for piece in pieces:
        if piece in special_tokens:
            word_counter[piece] += 1
        else:
            for m in re.finditer(PAT, piece):
                w = m.group(0)
                word_counter[w] += 1

    return word_counter


# ===============================
# BPE 主函数
# ===============================
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = None,
    verbose: bool = True,
):

    if num_processes is None:
        num_processes = cpu_count()

    # ---------- init vocab ----------
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    vocab_rev: Dict[bytes, int] = {v: k for k, v in vocab.items()}

    for tok in special_tokens:
        b = tok.encode("utf-8")
        idx = len(vocab)
        vocab[idx] = b
        vocab_rev[b] = idx

    num_merges = vocab_size - len(vocab)
    merges: List[Tuple[bytes, bytes]] = []

    # ---------- 预处理 ----------
    start_time = time.time()

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            num_processes,
            special_tokens[0].encode() if special_tokens else b"",
        )

        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    with Pool(num_processes) as pool:
        results = pool.map(
            process_chunk,
            [(chunk, special_tokens) for chunk in chunks],
        )

    word_counter = Counter()

    for wc in results:
        word_counter.update(wc)

    word_encode = {
        w: list(w.encode()) if w not in special_tokens else [vocab_rev[w.encode()]]
        for w in word_counter
    }

    if verbose:
        print(f"Preprocessing took {time.time() - start_time:.2f}s")

    # ---------- 构建 pair → words 索引 ----------
    pair_to_words = defaultdict(set)
    pair_counter = Counter()

    for w, freq in word_counter.items():
        tokens = word_encode[w]
        for a, b in zip(tokens[:-1], tokens[1:]):
            pair = (a, b)
            pair_counter[pair] += freq
            pair_to_words[pair].add(w)

    # ---------- BPE ----------
    start_time = time.time()

    for _ in trange(num_merges, disable=not verbose):

        if not pair_counter:
            break

        best_pair = max(
            pair_counter,
            key=lambda p: (pair_counter[p], vocab[p[0]], vocab[p[1]]),
        )

        new_id = len(vocab)
        new_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[new_id] = new_bytes
        vocab_rev[new_bytes] = new_id
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        affected_words = pair_to_words[best_pair].copy()

        # 清除旧 pair 计数
        del pair_counter[best_pair]
        del pair_to_words[best_pair]

        for w in affected_words:
            old_tokens = word_encode[w]
            freq = word_counter[w]

            # 移除旧 pair 统计
            for a, b in zip(old_tokens[:-1], old_tokens[1:]):
                pair = (a, b)
                pair_counter[pair] -= freq
                pair_to_words[pair].discard(w)

                if pair_counter[pair] <= 0:
                    del pair_counter[pair]
                    del pair_to_words[pair]

            # 进行 merge
            new_tokens = []
            i = 0
            while i < len(old_tokens):
                if (
                    i < len(old_tokens) - 1
                    and old_tokens[i] == best_pair[0]
                    and old_tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(old_tokens[i])
                    i += 1

            word_encode[w] = new_tokens

            # 添加新 pair 统计
            for a, b in zip(new_tokens[:-1], new_tokens[1:]):
                pair = (a, b)
                pair_counter[pair] += freq
                pair_to_words[pair].add(w)

    if verbose:
        print(f"BPE training took {time.time() - start_time:.2f}s")

    return vocab, merges


# ===============================
# 运行
# ===============================
if __name__ == "__main__":
    vocab, merges = train_bpe(
        "/Users/dym/Desktop/Project/assignment1-basics/data/test.txt",
        280,
        ["<|endoftext|>"],
        num_processes=4,
    )

    print("Total merges:", len(merges))
    print(merges)
