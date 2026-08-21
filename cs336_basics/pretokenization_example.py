import os
from typing import BinaryIO
import regex as re
from collections import Counter, defaultdict
from typing import Iterable

def test_find_chunk_boundaries():
    path = "data/debug.txt"
    with open(path, "rb") as f:
        print(find_chunk_boundaries(f, 4, ['<endoftext1>', '<endoftext2>']))
        assert find_chunk_boundaries(f, 4, ['<endoftext1>','<endoftext2>']) == [0, 20, 36, 52]

def test_split_on_special_token():
    path = "data/debug.txt"
    with open(path, "r") as f:
        print(list(_split_on_special_token(f.read(), ['<endoftext1>', '<endoftext2>'])))
        # assert list(_split_on_special_token(f.read(), ['<endoftext1>', '<endoftext2>'])) == ['aaaa', 'bbbb', 'cccc', 'dddd']


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    special_tokens: list[str],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        _bfind = False
        while _bfind == False:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            temp = -1
            for special_token in special_tokens:
                found_at = mini_chunk.find(special_token.encode("utf-8"))
                if found_at != -1:                                
                    if temp == -1 or found_at < temp:
                        temp = found_at
                    _bfind = True                
            if temp != -1:
                chunk_boundaries[bi] = initial_position + temp
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _split_on_special_token(text: str, special_tokens: list[str]) -> Iterable[str]:
    if not special_tokens:
        yield text
        return
    pat = re.compile("|".join(re.escape(token) for token in special_tokens))

    for piece in pat.split(text):
        if piece:
            yield piece

def pre_tokenize(chunk : str, special_tokens : list[str]) -> Counter[bytes]:
    PAT_gpt2 = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    counts: Counter[bytes] = Counter()

    for piece in _split_on_special_token(chunk, special_tokens):
        for match in PAT_gpt2.finditer(piece):
            pre_token = match.group(0).encode('utf-8')
            counts[pre_token] += 1

    return counts

# test_find_chunk_boundaries()
# test_split_on_special_token()