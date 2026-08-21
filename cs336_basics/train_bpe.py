import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries, pre_tokenize
from concurrent.futures import ProcessPoolExecutor as ppe
from typing import Iterable
from collections import Counter, defaultdict
import heapq
import time
Token = bytes
Pair = tuple[Token, Token]
Word = list[Token]
import os

NUM_PROC = 8


def train_bpe(input_path : str | os.PathLike,
vocab_size : int,
special_tokens: list[str]) -> tuple[dict[int, bytes],
list[tuple[bytes, bytes]]]:

    # initialize vocab
    vocab = {i : bytes([i]) for i in range(256)}
    next_token_id = len(vocab)

    special_tokens = sorted(set(special_tokens), key=len, reverse=True)
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1
    
    merges : list[Pair] = []

    time_pretokenize_begin = time.time()
    total_counts : Counter[bytes] = Counter()
    # split input into chunks to parallel pretokenization.
    with open(input_path, "rb") as f:
        num_processes = NUM_PROC
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens)


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
    time_pretokenize_end = time.time()


    time_merge_begin = time.time()
    # merge
    # count on pairs
    total_count_pairs : Counter[Pair] = Counter()
    for token, count in total_counts.items():
        if len(token) < 2:
            continue
        else:
            for i in range(len(token) - 1):
                total_count_pairs[(Token([token[i]]),Token([token[i + 1]]))] += count

    total_count_tuple : Counter[tuple[Token, ...]] = Counter()
    for token, count in total_counts.items():
        bis: tuple[Token, ...] = tuple(Token([bi]) for bi in token)
        total_count_tuple[bis] += count

    time_select = 0
    time_merge = 0
    time_recount = 0
    while len(vocab) < vocab_size and  len(total_count_pairs.items()) > 0:
        time_select1 = time.time()
        most_pair, most_count = max(total_count_pairs.items(), key=lambda x: (x[1], x[0]))
        time_select2 = time.time()
        time_select += time_select2 - time_select1
        vocab[next_token_id] = most_pair[0] + most_pair[1]
        next_token_id += 1
        merges.append(most_pair)
        # TODO according to mast_pair update total_count_tuple
        time_merge1 = time.time()
        new_counter : Counter[tuple[Token, ...]] = Counter()
        new_total_count_pairs : Counter[Pair] = Counter()
        for token, count in total_count_tuple.items():
            # merge pairs inside a pretoken
            i = 0
            new_token = []
            while i < len(token):
                if i + 1 < len(token) and (token[i], token[i + 1]) == most_pair:
                    new_token.append(token[i] + token[i + 1])
                    
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            new_counter[tuple(new_token)] += count

            i = 0
            while i + 1 < len(new_token):
                new_total_count_pairs[(new_token[i], new_token[i + 1])] += count
                i += 1
            total_count_pairs = new_total_count_pairs
                
        total_count_tuple = new_counter
        time_merge2 = time.time()
        time_merge += time_merge2 - time_merge1

        # time_recount1 = time.time()
        # total_count_pairs = Counter()
        # # TODO update total_count_pairs accordingly  x
        # # TODO reconstruct total_count_pairs 
        # for token, count in total_count_tuple.items():
        #     i = 0
        #     while i < len(token):
        #         if i + 1 < len(token):
        #             total_count_pairs[(token[i], token[i + 1])] += count
        #         i += 1
        # time_recount2 = time.time()
        # time_recount += time_recount2 - time_recount1
    time_merge_end = time.time()

    print(f"split time: {time_pretokenize_end - time_pretokenize_begin} s, merge time: {time_merge_end - time_merge_begin}")
    print(f"!!! detailed time: select = {time_select} \n merge = {time_merge} \n recount = {time_recount}")
    # TODO return final result
    return (vocab, merges)

# train_bpe("data/debug.txt", 266, ['<endoftext1>', '<endoftext2>'])