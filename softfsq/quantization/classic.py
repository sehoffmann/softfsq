from typing import override

import einops
import torch
from torch import nn

from softfsq.losses import LossMixin
from softfsq.quantization.common import QuantizedTensors, Quantizer

__all__ = [
    'VectorQuantizer',
]

class LpNormalization(nn.Module):
    """
    Module to perform Lp normalization on input tensors.
    See: torch.nn.functional.normalize()
    """

    def __init__(self, p: float = 2, dim: int = 1, eps: float = 1e-12):
        """
        Args:
            p (float): The exponent value in the Lp norm. Default is 2.
            dim (int): The dimension along which to compute the norm. Default is 1.
            eps (float): A small value to avoid division by zero. Default is 1e-12.
        """
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x, p=self.p, dim=self.dim, eps=self.eps)



@torch.no_grad()
def nearest_neighbor(x, codebook):
    """
    Find the nearest neighbor in the codebook for each vector in x.

    Args:
        x (torch.Tensor): Input tensor of shape (..., D).
        codebook (torch.Tensor): Codebook tensor of shape (N, D).

    Returns:
        torch.Tensor: Indices of the nearest neighbors in the codebook, shape (...,).
    """
    x_flat = x.flatten(start_dim=0, end_dim=-2)  # flatten to (M, D)
    d = torch.cdist(x_flat.unsqueeze(0), codebook.unsqueeze(0))  # distances as (1, M, N)
    indices = torch.argmin(d, dim=2).squeeze(0)  # (M,)
    indices = indices.reshape(x.shape[:-1])  # reshape to original shape without D
    return indices


class VectorQuantizer(LossMixin, Quantizer):
    """
    Classic Vector Quantization from VQ-VAE with learned codebook.
    Optional with normalization of inputs and codebook vectors.
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        initialization: str = 'normal',
        normalization: str = 'layer_norm',
        commitment_weight: float = 0.1,
        embedding_weight: float = 0.25,
    ):
        super().__init__()
        self.commitment_weight = commitment_weight
        self.embedding_weight = embedding_weight

        if normalization == 'none':
            self.norm = nn.Identity()
        elif normalization == 'l1':
            self.norm = LpNormalization(p=1, dim=-1)
        elif normalization == 'l2':
            self.norm = LpNormalization(p=2, dim=-1)
        elif normalization == 'linf':
            self.norm = LpNormalization(p=float('inf'), dim=-1)
        elif normalization == 'layer_norm':
            self.norm = nn.LayerNorm(codebook_dim, elementwise_affine=False)
        else:
            raise ValueError('Normalization must be one of ["none", "l1", "l2", "linf", "layer_norm"]')

        if initialization not in ['normal', 'uniform']:
            raise ValueError('Initialization must be one of ["normal", "uniform"]')
        self.initialization = initialization

        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        self.reset_parameters()

    def reset_parameters(self):
        if self.initialization == 'uniform':
            self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
        elif self.initialization == 'normal':
            self.embedding.weight.data.normal_()
        else:
            assert False, 'unreachable'

    @override
    @property
    def codebook_size(self) -> int:
        return self.embedding.num_embeddings

    @override
    @property
    def codebook_dim(self) -> int:
        return self.embedding.embedding_dim

    @override
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        y = self.embedding(indices)  # (B, ..., D)
        y = einops.rearrange(y, 'b ... d -> b d ...')
        return y.contiguous()

    @override
    def encode(self, inputs):
        inputs = einops.rearrange(inputs, 'b c ... -> b ... c').contiguous()  # channels last
        z_normed = self.norm(inputs)
        codebook_normed = self.norm(self.embedding.weight)

        # quantize
        indices = nearest_neighbor(z_normed, codebook_normed)  # (B, ...)
        z_q = codebook_normed[indices]  # (B, ..., D)

        # to channels first
        z_q = einops.rearrange(z_q, 'b ... c -> b c ...').contiguous()
        z_normed = einops.rearrange(z_normed, 'b ... c -> b c ...').contiguous()

        # compute loss
        commitment_loss = self.commitment_weight * torch.mean((z_q.detach() - z_normed) ** 2)
        embedding_loss = self.embedding_weight * torch.mean((z_q - z_normed.detach()) ** 2)
        self.set_loss('commitment_loss', commitment_loss)
        self.set_loss('embedding_loss', embedding_loss)

        z_q_str = z_normed + (z_q - z_normed).detach()  # straight-through estimator
        return QuantizedTensors(values=z_q_str, indices=indices, pre_quantization=z_normed)
