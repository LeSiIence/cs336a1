from cs336_basics.train_bpe import train_bpe
import pickle
vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])

with open("model/bpe_model_tinystories.pkl", "wb") as f:
    pickle.dump({"vocab": vocab, "merges": merges}, f)

print("BPE saved!")

