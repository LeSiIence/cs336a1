import torch
import torch.nn as nn
import einops
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        W = torch.empty(
            (d_model),
            device=device,
            dtype=dtype
        )
        self.weight = nn.Parameter(W)
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_sq = einops.reduce(x ** 2, "... d -> ... 1", "mean")
        rms = torch.sqrt(mean_sq + self.eps)
        result = (x / rms) * self.weight
        result = result.to(in_dtype)
        return result