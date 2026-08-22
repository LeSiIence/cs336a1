from cs336_basics.train_bpe import train_bpe
import pickle
vocab, merges = train_bpe("data/owt_valid.txt", 32000)

with open("bpe_model_owt.pkl", "wb") as f:
    pickle.dump({"vocab": vocab, "merges": merges}, f)

print("BPE saved!")
