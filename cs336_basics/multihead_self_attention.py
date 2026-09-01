import torch
import torch.nn as nn
from cs336_basics.rope import RoPE
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from jaxtyping import Float, Int
import einops

class Multihead_self_attention(nn.Module):
    def __init__(self, d_model, num_heads, use_rope = False, max_seq_len=2048, theta=10000.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE(d_k=self.d_k, max_seq_len=max_seq_len, theta=theta)
        else:
            self.rope = None
    def forward(self, q_proj_weight: Float[torch.Tensor, " d_model d_model"],
        k_proj_weight: Float[torch.Tensor, " d_model d_model"],
        v_proj_weight: Float[torch.Tensor, " d_model d_model"],
        o_proj_weight: Float[torch.Tensor, " d_model d_model"],
        in_features: Float[torch.Tensor, " ... sequence_length d_model"],
        token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None
        )-> Float[torch.Tensor, " ... sequence_length d_model"]:
        seq_len = in_features.shape[-2]

        qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
        qkv = einops.einsum(in_features, qkv_weight, "... seq d_in, d_out d_in -> ... seq d_out")

        q, k, v = einops.rearrange(
            qkv,
            "... seq (split heads d_k) -> split ... heads seq d_k",
            split = 3,
            heads = self.num_heads
        )
        if self.use_rope and self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=in_features.device))
        attention_out = scaled_dot_product_attention(q, k, v, mask=mask)

        attention_out = einops.rearrange(attention_out, "... heads seq d_v -> ... seq (heads d_v)")
        out = einops.einsum(attention_out, o_proj_weight, "... seq d_in, d_out d_in -> ... seq d_out")

        return out