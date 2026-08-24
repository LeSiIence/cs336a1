import pickle
import json
import regex as re
from typing import Iterable, Iterator
Token = bytes
Pair = tuple[Token, Token]
Word = list[Token]

class Tokenizer:
    
    def __init__(
        self,
        vocab: dict[int, Token],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab : dict[int, Token] = vocab
        self.merges : list[Pair] = merges
        self.special_tokens: list[str] = special_tokens or []
        self.special_tokens = sorted(
            self.special_tokens,
            key=len,
            reverse=True
        )
        self._inverse_vocab : dict[Token, int] = {
            token : token_id for token_id, token in vocab.items()
        }
        self._merge_rank : dict[Pair, int] = {
            pair : rank for rank, pair in enumerate(self.merges)
        }

        self.token_buffer: dict[Token, list[Token]] = {}

    @classmethod
    def from_files(cls, 
        vocab_filepath : str, # json format like {"!": 0}
        merges_filepath : str, # txt, needs handle, per line per pair
        special_tokens : list[str] = []):

        vocab = {}
        if vocab_filepath != "":
            with open(vocab_filepath, "r", encoding= "utf-8") as f:
                vocab_r : dict[str, int] = json.load(f)
            vocab = {
                token_id : token.encode("utf-8")
                for token, token_id in vocab_r.items()
            }

        merges : list[Pair] = []

        if merges_filepath != "":
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
        # 1. split on special_tokens
        parts = self._split_on_special_token(text, self.special_tokens)
        pre_tokens = []
        for part in parts:
            if part in self.special_tokens:
                pre_tokens.append(part.encode("utf-8"))
            else:
                pre_tokens.extend(self._pre_tokenize(part))
        

        # 2.apply bpe merges
        tokens : list[Token] = []
        
        for pre_token in pre_tokens:
            if pre_token.decode("utf-8") in self.special_tokens:                
                tokens.append(pre_token)
                continue

            if pre_token not in self.token_buffer:
                sub_tokens = self._apply_bpe(pre_token)
                self.token_buffer[pre_token] = sub_tokens                
            else:
                sub_tokens = self.token_buffer[pre_token]
            tokens.extend(sub_tokens)

        encoded : list[int] = [self._inverse_vocab[token] for token in tokens]

        return encoded

    def encode_iterable(self, iterable : Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # text : str = ""
        # for id in ids:
        #     token = self.vocab[id]
        #     text += token.decode("utf-8")
        # !!! fixed fatal error above: some multi-byte characters get across two tokens
        
        # should decode at a time before returning !!!
        btext : bytes = b''
        for id in ids:
            token = self.vocab[id]
            btext += token
        text = btext.decode("utf-8", errors="replace")
        return text

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
        PAT_gpt2 = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        pre_tokens : list[Token] = []
        for match in PAT_gpt2.finditer(text):
            pre_token = match.group(0).encode("utf-8")
            pre_tokens.append(pre_token)

        return pre_tokens

    def _apply_bpe(self, pre_token : Token) -> list[Token]:
        
        old_token : list[Token] = [Token([bis]) for bis in pre_token]

        while True:                   
            # select pair
            most_pair = None
            for i in range(len(old_token) - 1):
                p : Pair = (old_token[i], old_token[i + 1])
                # if pre_tokenmost_pair is None or (p in self._merge_rank and self._merge_rank[p] < self._merge_rank[most_pair]):
                if p in self._merge_rank and (most_pair is None or self._merge_rank[p] <= self._merge_rank[most_pair]):
                    most_pair = p
            if most_pair is None:
                break

            # merge pair & update pretoken
            new_token : list[Token] = []
            n = len(old_token)
            i = 0
            while i < n:
                if i + 1 < n and (old_token[i], old_token[i + 1]) == most_pair:
                    new_token.append(old_token[i] + old_token[i + 1])
                    i += 2
                else:
                    new_token.append(Token(old_token[i]))
                    i += 1

            old_token = new_token

        return old_token

    def _split_on_special_token(self, text: str, special_tokens: list[str]) -> Iterable[str]:
        if not special_tokens:
            yield text
            return
        # pat = re.compile("|".join(re.escape(token) for token in special_tokens))
        pat = re.compile(
        "(" + "|".join(
            re.escape(token)
            for token in special_tokens
        ) + ")"
        )
        for piece in pat.split(text):
            if piece:
                yield piece

# def test_apply_bpe():
#     t = Tokenizer(vocab = {0 : b'a', 1 :b'b', 2 : b'aa'}, merges=[(b'a', b'a')])
#     print(t._apply_bpe(b"aabaaaa"))

# test_apply_bpe()

# def test_encode():
#     t = Tokenizer(vocab = {0 : b'a', 1 :b'b', 2 : b'aa'}, merges=[(b'a', b'a')])
#     print(t.encode("aabaaaa"))
# test_encode()

# def test_decode():
#     t = Tokenizer(vocab = {0 : b'a', 1 :b'b', 2 : b'aa'}, merges=[(b'a', b'a')])
#     print(t.decode([2, 1, 2, 2]))     
# test_decode()