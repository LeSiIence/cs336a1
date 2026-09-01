import torch
import torch.nn as nn
import einops

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        dim_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (dim_indices / d_k))
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = einops.einsum(positions, inv_freq, "i, j -> i j")
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_cached: torch.Tensor = self.cos_cached
        sin_cached: torch.Tensor = self.sin_cached
        cos = cos_cached[token_positions].to(device=x.device, dtype=x.dtype)
        sin = sin_cached[token_positions].to(device=x.device, dtype=x.dtype)

        x_pairs = einops.rearrange(x, "... (d two) -> ... d two", two=2)
        x_even = x_pairs[..., 0]
        x_odd  = x_pairs[..., 1]

        rot_even = x_even * cos - x_odd * sin
        rot_odd  = x_even * sin + x_odd * cos

        rotated_pairs = torch.stack([rot_even, rot_odd], dim=-1)
        out = einops.rearrange(rotated_pairs, "... d two -> ... (d two)")
        return out