import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries, pre_tokenize
from concurrent.futures import ProcessPoolExecutor as ppe
from typing import Iterable
from collections import Counter, defaultdict
import heapq
import time
from tqdm import tqdm


Token = bytes
Pair = tuple[Token, Token]
Word = list[Token]
import os




def train_bpe(input_path : str | os.PathLike,
vocab_size : int,
special_tokens: list[str] = [], 
num_processes : int = 16) -> tuple[dict[int, bytes],
list[tuple[bytes, bytes]]]:

    # 1. initialize vocab
    vocab = {i : bytes([i]) for i in range(256)}
    next_token_id = len(vocab)

    special_tokens = sorted(set(special_tokens), key=len, reverse=True)
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1
    
    merges : list[Pair] = []

    time_pretokenize_begin = time.time()
    total_counts : Counter[bytes] = Counter()
    # 2. split input into chunks to parallel pretokenization.
    with open(input_path, "rb") as f:
        
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens)


        # parallel pretokenize each chunk with multiple processes.
        with ppe(max_workers=num_processes) as t:
            futures = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                future = t.submit(pre_tokenize, chunk, special_tokens)
                futures.append(future)

            for future in tqdm(futures, desc="Chunk pretokenization", unit="chunk"):
                total_counts.update(future.result())
    time_pretokenize_end = time.time()


    time_merge_begin = time.time()
    merge_total = max(0, vocab_size - len(vocab))
    merge_progress = tqdm(total=merge_total, desc="BPE merge", unit="merge")
    # 3. merge
    # count on pairs
    total_count_pairs : Counter[Pair] = Counter()
    for token, count in total_counts.items():
        for i in range(len(token) - 1):
            total_count_pairs[(Token([token[i]]),Token([token[i + 1]]))] += count

    # count on pretokens
    total_count_tuple : Counter[tuple[Token, ...]] = Counter()
    for token, count in total_counts.items():
        bis: tuple[Token, ...] = tuple(Token([bi]) for bi in token)
        total_count_tuple[bis] += count

    # initialize affected pretokens
    pair_affected_pretokens : dict[Pair, set[tuple[Token, ...]]] = defaultdict(set)
    for token in total_count_tuple:
        for i in range(len(token) - 1):
            pair_affected_pretokens[(token[i], token[i + 1])].add(token) 

    def make_heap_entry(pair: Pair, freq: int):
        neg_p0 = bytes(255 - b for b in pair[0])
        neg_p1 = bytes(255 - b for b in pair[1])
        return (-freq, neg_p0, neg_p1, pair)

    # initialize heap
    pair_heap: list[tuple[int, bytes, bytes, Pair]] = [
        make_heap_entry(p, c) for p, c in total_count_pairs.items()
    ]
    heapq.heapify(pair_heap)

    time_select = 0
    time_merge = 0
    time_recount = 0
    while len(vocab) < vocab_size and  len(total_count_pairs.items()) > 0:
        time_select1 = time.time()
        #most_pair, most_count = max(total_count_pairs.items(), key=lambda x: (x[1], x[0]))
        # TODO refactor: use heap to select max pair
        most_pair = None
        while pair_heap:
            neg_freq, _, _, p = heapq.heappop(pair_heap)
            curr_freq = total_count_pairs.get(p, 0)
            # 只有当堆顶频次与当前真实频次完全一致时，才是有效数据
            if curr_freq > 0 and -neg_freq == curr_freq:
                most_pair = p
                break

        if most_pair is None:
            break

        time_select2 = time.time()
        time_select += time_select2 - time_select1
        vocab[next_token_id] = most_pair[0] + most_pair[1]
        next_token_id += 1
        merges.append(most_pair)
        merge_progress.update(1)

        # according to mast_pair update total_count_tuple
        time_merge1 = time.time()
        # new_counter_tuple : Counter[tuple[Token, ...]] = Counter()
        # new_total_count_pairs : Counter[Pair] = Counter()

        affected_pretokens = list(pair_affected_pretokens.pop(most_pair, set()))
        
        # better update of total_count_tuple using pair->pretoken dict
        # for token, count in total_count_tuple.items():
        for token in affected_pretokens:
            count = total_count_tuple.pop(token, 0)
            if count == 0:
                continue
            
            # update index
            for i in range(len(token) - 1):
                p = (token[i], token[i + 1])
                total_count_pairs[p] -= count
                if total_count_pairs[p] == 0:
                    del total_count_pairs[p]
                elif total_count_pairs[p] < 0:
                    raise ValueError
                pair_affected_pretokens[p].discard(token)

            # merge pairs inside a pretoken
            new_token = []
            i = 0
            n = len(token)
            while i < n:
                if i + 1 < n and (token[i], token[i + 1]) == most_pair:
                    new_token.append(token[i] + token[i + 1])
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            new_token = tuple(new_token)

            # add new_token to total_count_tuple
            total_count_tuple[new_token] = total_count_tuple.get(new_token, 0) + count
            for i in range(len(new_token) - 1):
                p = (new_token[i], new_token[i + 1])
                total_count_pairs[p] += count
                pair_affected_pretokens[p].add(new_token)

      
        time_merge2 = time.time()
        time_merge += time_merge2 - time_merge1

        # time_recount1 = time.time()
        # total_count_pairs = Counter()
        # # reconstruct total_count_pairs 
        # for token, count in total_count_tuple.items():
        #     i = 0
        #     while i < len(token):
        #         if i + 1 < len(token):
        #             total_count_pairs[(token[i], token[i + 1])] += count
        #         i += 1
        # time_recount2 = time.time()
        # time_recount += time_recount2 - time_recount1
    merge_progress.close()
    time_merge_end = time.time()

    print(f"split time: {time_pretokenize_end - time_pretokenize_begin} s, merge time: {time_merge_end - time_merge_begin}")
    print(f"!!! detailed time: select = {time_select} \n merge = {time_merge} \n recount = {time_recount}")
    # return final result
    return (vocab, merges)

# train_bpe("data/debug.txt", 266, ['<endoftext1>', '<endoftext2>'])