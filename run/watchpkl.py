import pickle
import json

with open("bpe_model_owt.pkl", "rb") as f:
    obj = pickle.load(f)

vocab = obj["vocab"]
merges = obj["merges"]

payload = {
    "vocab": {str(k): v.decode("latin1") for k, v in vocab.items()},
    "merges": [[a.decode("latin1"), b.decode("latin1")] for a, b in merges]
}

longest_token_id, longest_token = max(vocab.items(), key=lambda kv: len(kv[1]))
print(
    "longest token:",
    {
        "id": longest_token_id,
        "token": longest_token.decode("latin1"),
        "byte_length": len(longest_token),
    },
)

with open("bpe_model.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

    
