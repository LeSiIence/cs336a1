import torch
import torch.nn as nn
import einops
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        W = torch.empty(
            (out_features, in_features),
            device=device,
            dtype=dtype
        )
        self.weight = nn.Parameter(W)
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")