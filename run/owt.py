from cs336_basics.train_bpe import train_bpe
import pickle
vocab, merges = train_bpe("data/owt_train.txt", 32000, ["<|endoftext|>"], num_processes=8)

with open("model/bpe_model_owt.pkl", "wb") as f:
    pickle.dump({"vocab": vocab, "merges": merges}, f)

print("BPE saved!")
