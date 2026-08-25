from itertools import islice
from cs336_basics.bpe_encoding import Tokenizer
import random
import time
owt_pkl_path = "model/bpe_model_owt.pkl"
owt_text_path = "data/owt_valid.txt"
tiny_pkl_path = "model/bpe_model_tinystories.pkl"
tiny_text_path = "data/TinyStoriesV2-GPT4-valid.txt"


def sample(pkl_path : str, text_path : str, k : int = 10):
    special = ["<|endoftext|>"]
    t = Tokenizer._from_single_file_pkl(pkl_path)
    with open(text_path, "r") as f:
        text = f.read()
    pieces = [x for x in t._split_on_special_token(text) if x != "<|endoftext|>"]

    encoded = []
    len_text = 0
    random.seed(2026)
    time0 = time.time()

    for piece in random.sample(pieces, k=k) if k > 0 else pieces:
        if piece == "<|endoftext|>":
            raise ValueError
        encoded_piece = t.encode(piece)
        encoded.extend(encoded_piece)
        len_text += len(piece.encode("utf-8"))
    
    time1 = time.time()

    throughput = len_text / (time1 - time0)
    compress_ratio = len_text / len(encoded) if encoded else 0.0
    print(f"compress_ratio = {compress_ratio}, throughput = {throughput}")
    return {compress_ratio, throughput}

#a
sample(owt_pkl_path, owt_text_path)
sample(tiny_pkl_path, tiny_text_path)
#b
sample(tiny_pkl_path, owt_text_path)

#c
sample(owt_pkl_path, owt_text_path, k=-1)
