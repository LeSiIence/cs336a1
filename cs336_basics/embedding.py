import torch
import torch.nn as nn
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module. This function should accept the following parameters:
        num_embedding: int: Size of the vocabulary
        embedding_dim: int: Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None: Device to store the parameters on
        dtype: torch.dtype | None = None: Data type of the parameters
        """
        super().__init__()
        weight = torch.empty(
            (num_embeddings, embedding_dim),
            device=device,
            dtype=dtype
        )
        self.weight = nn.Parameter(weight)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs

        """
        return self.weight[token_ids]
