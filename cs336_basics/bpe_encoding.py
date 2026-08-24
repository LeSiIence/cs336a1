import os
import pickle
import json
from cs336_basics.pretokenization_example import find_chunk_boundaries, pre_tokenize
from typing import overload, Iterable, Iterator
Token = bytes
Pair = tuple[Token, Token]
Word = list[Token]

class Tokenizer:
    
    def __init__(self, vocab : dict[int, Token], merges : list[Pair], special_tokens) -> None:
        self.vocab : dict[int, Token] = vocab
        self.merges : list[Pair] = merges
        self.special_tokens : list[str] | None = special_tokens
        if not special_tokens:
            self.special_tokens = []
        self._inverse_vocab : dict[Token, int] = {
            token : token_id for token_id, token in vocab.items()
        }
        self._merge_rank : dict[Pair, int] = {
            pair : rank for rank, pair in enumerate(self.merges)
        }

    @classmethod
    def from_files(cls, 
        vocab_filepath : str, # json format like {"!": 0}
        merges_filepath : str, # txt, needs handle, per line per pair
        special_tokens : list[str] = []):
        with open(vocab_filepath, "r", encoding= "utf-8") as f:
            vocab_r : dict[str, int] = json.load(f)
        vocab = {
            token_id : token.encode("utf-8")
            for token, token_id in vocab_r.items()
        }

        merges : list[Pair] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid merge line: {line!r}")
                merges.append(
                    (
                        parts[0].encode("utf-8"),
                        parts[1].encode("utf-8"),
                    )
                )

        return cls(vocab, merges, special_tokens)

    def encode(self, text : str) -> list[int]:
        # TODO
        return []

    def encode_iterable(self, iterable : Iterable[str]) -> Iterator[int]:
        # TODO
        yield 0

    def decode(self, ids: list[int]) -> str:
        # TODO
        return ""

    @classmethod
    def _from_single_file_pkl(cls, 
        file_path : str, 
        special_tokens : list[str] = []):
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        vocab = obj["vocab"]
        merges = obj["merges"]
        return cls(vocab, merges, special_tokens)
        
    
    def _pre_tokenize(self, text : str) -> list[Token]: 
        # TODO 
        return []

    def _apply_bpe(self, pretoken : Token) -> list[Token]:
        return []
    
    


    

    


    






        

