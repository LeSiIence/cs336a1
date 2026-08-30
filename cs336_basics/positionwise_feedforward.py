import torch
import torch.nn as nn
import einops
from .linear import Linear       

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.l1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.l2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.l3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.l1(x)
        w3x = self.l3(x)
        silu_w1x = w1x / (1 + torch.exp(-w1x))
        return self.l2(silu_w1x * w3x) 