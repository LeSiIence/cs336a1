import torch
from jaxtyping import Bool, Float, Int
import einops
from cs336_basics.softmax import softmax
def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... keys d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scores = einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, -float("inf"))
    
    attention_weights = softmax(scores, dim=-1)
    return einops.einsum(attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
    