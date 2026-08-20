
import regex as re
from .pretokenization_example import find_chunk_boundaries, pre_tokenize
from concurrent.futures import ProcessPoolExecutor as ppe
from typing import Iterable
from collections import Counter, defaultdict
Token = bytes
Pair = tuple[Token, Token]
Word = list[Token]

NUM_PROC = 8


def train_bpe(input_path : str, vocab_size : int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # initialize vocab
    vocab = {i : bytes([i]) for i in range(256)}
    next_token_id = len(vocab)

    special_tokens = sorted(set(special_tokens), key=len, reverse=True)
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1
    
    merges : list[Pair] = []

    total_counts : Counter[bytes] = Counter()
    # split input into chunks to parallel pretokenization.
    with open(input_path, "rb") as f:
        num_processes = NUM_PROC
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")


        # parallel pretokenize each chunk with multiple processes.
        with ppe(max_workers=NUM_PROC) as t:
            futures = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                future = t.submit(pre_tokenize, chunk, special_tokens)
                futures.append(future)

            for future in futures:
                total_counts.update(future.result())

    return ({}, [])
