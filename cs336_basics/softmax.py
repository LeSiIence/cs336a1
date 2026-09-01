import torch
def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    temp = in_features
    max_term = torch.max(temp, dim=dim, keepdim=True).values
    temp = temp - max_term
    exp = torch.exp(temp)
    return exp / exp.sum(dim=dim, keepdim=True)

