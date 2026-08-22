# cs336 assignment 1

## Sec.2 BPE Tokenizer

### 2.1

- (a) What Unicode character does chr(0) return? 

  a non-printable control character whose official name is `NULL`.

- (b)How does this character’s string representation (`__repr__()`) differ from its printed representation?

  `\x00` and ` `

- (c)What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

  (c) When U+0000 occurs in text, it remains part of the string but is normally invisible when printed(except for some frontend rendering), so the surrounding visible characters appear adjacent.

  ```bash
  'this is a test\x00string'
  >>> chr(0)
  '\x00'
  >>> print(chr(0))
  
  >>> "this is a test" + chr(0) + "string"
  'this is a test\x00string'
  >>> print("this is a test" + chr(0) + "string")
  this is a teststring
  ```

### 2.2

- (a)What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

  Because most of the training text, which is 98%+ of the web pages are encoded in utf-8. Besides, using utf-8 is more compact for text only in ascii characters.

- (b)Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

  ```python
  def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
  return "".join([bytes([b]).decode("utf-8") for b in bytestring])
  ```

  

  utf-8 bytestring cannot be interpreted byte by byte for there are characters taking more than one byte. in the case below, those string with characters of more than one byte, will never be interpreted correctly. For example, `"hello! こんにちは!"`

- (c)Give a two-byte sequence that does not decode to any Unicode character(s).

  `b'\xff\xff'`, for the start of 1-byte char is `0xxxxxxx`, for 2-bytes:`110xxxxx`, for 3-bytes:`1110xxxx` and for 4-bytes:`11110xxx` and the following bytes must be `10xxxxxx`

### 2.3

### 2.4

### 2.5 

- Problem (train_bpe): BPE Tokenizer Training (15 points)

  ```python
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
  ```

- `commit: 56c59ae8d33e652521ddd2930f1251f46d8c303c`

  Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

  (a)Elapsed (wall clock) time (h:mm:ss or m:ss): 1:04.33

  Maximum resident set size (kbytes): 7635972

  longest token: {'id': 7160, 'token': ' accomplishment', 'byte_length': 15} is a real English word and it is very common to use in training txt

  (b)Using time.time(), the detailed time use is:

  ```txt
  pretokenize time: 42.98758864402771 s, merge time: 21.021564722061157
  
  detailed merge time: select = 19.14059090614319 
  
  merge = 1.6145775318145752 
  ```

  The pre-tokenization phase takes the most time overall, accounting for roughly 43 seconds (about 67%) of the total runtime. Within the BPE merge loop itself, selecting the most frequent pair via `max()` dominates the remaining time (taking 19.14 seconds).

- `commit: 49fb76ad8ac3d2e8d0e72ff32ef33ecaffa35ca7`(use lazy heap)

  Elapsed (wall clock) time (h:mm:ss or m:ss): 0:47.63

  Maximum resident set size (kbytes): 7634112
  
  longest token: {'id': 7160, 'token': ' accomplishment', 'byte_length': 15} is a real English word and it is very common to use in training txt
  
  (b)Using time.time(), the detailed time use is:
  
  pretokenize time: 44.7478723526001 s, merge time: 2.7014994621276855
  detailed time: select = 0.38849925994873047 
   merge = 2.0813822746276855
  
  The pre-tokenization phase takes the most time overall, accounting for roughly 45 seconds of the total runtime. Within the BPE merge loop itself, selecting the most frequent pair via `max()` no longer dominates the remaining time (taking 0.0018 seconds).
  
- Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
  (a)
  Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?
  Resource requirements: ≤12 hours (no GPUs), ≤100 GB RAM
  Deliverable: A one-to-two sentence response.
  
```
  longest token: {'id': 25822, 'token': 'Ã\x83Ã\x82Ã\x83Ã\x82Ã\x83Ã\x82Ã\x83Ã\x82Ã\x83Ã\x82Ã\x83Ã\x82Ã\x83Ã\x82Ã\x83Ã\x82Ã\x83Ã\x82Ã\x83Ã\x82Ã\x83Ã\x82Ã\x83Ã\x82Ã\x83Ã\x82Ã\x83Ã\x82Ã\x83Ã\x82Ã\x83Ã\x82', 'byte_length': 64} 
```

  it's strange but actually making sense for there might be some error in the text from the web. The longest token in the OpenWebText vocabulary is a 64-byte sequence consisting of repeated mojibake bytes corresponding to `ÃÂÃÂ...`. This makes sense because OpenWebText contains noisy web-scraped text with encoding artifacts, and BPE repeatedly merges frequent byte sequences regardless of whether they are semantically meaningful.

- (b)
  Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
  Deliverable: A one-to-two sentence response.

  ```
  (base) nanxin@Asus:~/workspace/cs336/assignment1-basics$ /usr/bin/time -v uv run python run/owt.py 2>&1 | tee bpe_output.log
  Chunk pretokenization: 100%|██████████| 8/8 [10:18<00:00, 77.31s/chunk] 
  BPE merge: 100%|██████████| 31743/31743 [15:43<00:00, 33.65merge/s]              split time: 1530.3117876052856 s, merge time: 943.4129085540771
  !!! detailed time: select = 62.855557441711426 
   merge = 812.7013056278229 
   recount = 0
  BPE saved!
          Command being timed: "uv run python run/owt.py"        User time (seconds): 2373.28
          System time (seconds): 523.89
          Percent of CPU this job got: 116%        Elapsed (wall clock) time (h:mm:ss or m:ss): 41:35.76
          Average shared text size (kbytes): 0        Average unshared data size (kbytes): 0
          Average stack size (kbytes): 0        Average total size (kbytes): 0
          Maximum resident set size (kbytes): 24868808
          Average resident set size (kbytes): 0
          Major (requiring I/O) page faults: 1853424
          Minor (reclaiming a frame) page faults: 46130618        Voluntary context switches: 2693522
          Involuntary context switches: 37451
          Swaps: 0        File system inputs: 116117640
          File system outputs: 1632
          Socket messages sent: 0
          Socket messages received: 0
          Signals delivered: 0
          Page size (bytes): 4096
          Exit status: 0
  ```
  
  
  
  















