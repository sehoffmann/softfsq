from typing import override, Sequence

import einops
import numpy as np
import torch

from softfsq.losses import LossMixin
from softfsq.math.distributions import MVNormalEstimator
from .common import QuantizedTensors, Quantizer


def entropy_to_sigma(entropy):
    return np.exp(entropy / np.log(2)) / np.sqrt(2 * np.pi * np.e)


def normal_kl_divergence(mu_1, sigma_1, mu_2, sigma_2):
    log_term = torch.log(sigma_2 / sigma_1)
    mse_term = (sigma_1**2 + (mu_1 - mu_2) ** 2) / (2 * sigma_2**2)
    return log_term + mse_term - 0.5


def standard_normal_kl_divergence(mu, sigma):
    log_term = -torch.log(sigma)
    mse_term = (sigma**2 + mu**2) / 2
    return log_term + mse_term - 0.5


def kl_divergence(mu, sigma, entropy):
    target_sigma = entropy_to_sigma(entropy)
    return normal_kl_divergence(mu, sigma, torch.zeros_like(mu), torch.full_like(sigma, target_sigma))


def variance_regularizer(sigma, entropy):
    target_sigma = entropy_to_sigma(entropy)
    return torch.log(sigma / target_sigma) + (target_sigma**2) / (2 * sigma**2) - 0.5


def normal_cdf(x: torch.Tensor):
    return 0.5 * (1 + torch.erf(x / np.sqrt(2)))


def normal_quantile(p: torch.Tensor):
    return np.sqrt(2) * torch.erfinv(2 * p - 1)


class AdaptiveNormalQuantization(LossMixin, Quantizer):
    def __init__(self, dim, num_bins: int | Sequence[int], lag: float | int = 10_000, kl_weight: float = 0.0):
        super().__init__()

        if not isinstance(num_bins, Sequence):
            num_bins = [num_bins] * dim

        if len(num_bins) != dim:
            raise ValueError('num_bins must have the same length as dim')

        if any(n < 2 for n in num_bins):
            raise ValueError('All elements of num_bins must be at least 2')

        self.dim = dim
        self.normal_estimator = MVNormalEstimator(dim=dim, lag=lag)

        self.register_buffer('num_bins', torch.tensor(num_bins, dtype=torch.int64))
        basis = torch.cumprod(torch.cat([torch.tensor([1], dtype=torch.int64), self.num_bins[:-1]]), dim=0)  # D
        self.register_buffer('basis', basis)

        bins = []
        centers = []
        for d in range(dim):
            quantiles = torch.linspace(0, 1, num_bins[d] + 1)[1:-1]  # exclude 0 and 1
            bins.append(quantiles)  # convert to standard normal quantiles
            half_width = (quantiles[1] - quantiles[0]) / 2
            center_quantiles = (
                torch.cat([quantiles, torch.tensor([1.0])]) - half_width
            )  # (n+1) to include the last bin center
            centers.append(center_quantiles)

        nt = torch.nested.nested_tensor(bins, layout=torch.jagged)
        self.register_buffer('bins', torch.nested.to_padded_tensor(nt, 0.0))

        nt = torch.nested.nested_tensor(centers, layout=torch.jagged)
        self.register_buffer('centers', torch.nested.to_padded_tensor(nt, 0.0))

        self.register_buffer('kl_weight', torch.tensor(kl_weight))

    @override
    @property
    def codebook_size(self) -> int:
        return torch.prod(self.num_bins).item()

    @override
    @property
    def codebook_dim(self) -> int:
        return self.dim

    @torch.no_grad()
    def _encode_indices(self, indices_md: torch.Tensor) -> torch.Tensor:
        # indices_md is of shape (..., D)
        # We need to convert it to a single index of shape (...)
        indices = torch.sum(indices_md * self.basis, dim=-1)  # broadcasted sum over D
        return indices

    @torch.no_grad()
    def _decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        # indices is of shape (B, ...)
        # We need to convert it to shape (B, ..., D)
        indices_md = []
        for d in range(self.dim):
            indices_d = (indices // self.basis[d]) % self.num_bins[d]
            indices_md.append(indices_d)
        return torch.stack(indices_md, dim=-1)

    @torch.no_grad()
    def _decode(self, indices: torch.Tensor, mean: torch.Tensor = None, C: torch.Tensor = None) -> torch.Tensor:
        # indices is of shape (B, ...)
        indices_md = self._decode_indices(indices)  # B x ... x D
        shape = indices_md.shape
        indices_flat = indices_md.reshape(-1, self.dim)  # N x D

        if mean is None or C is None:
            mean, cov = self.normal_dist.mean, self.normal_dist.covariance
            C = torch.linalg.cholesky(cov)  # D x D

        centers_normed = torch.take_along_dim(self.centers, indices_flat.permute(1, 0), dim=1).permute(1, 0)  # N x D
        quantized = centers_normed @ C.T + mean  # N x D
        quantized = quantized.reshape(*shape)  # B x ... x D
        return einops.rearrange(quantized, 'B ... D -> B D ...')

    @override
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self._decode(indices)

    @override
    def encode(self, x: torch.Tensor) -> QuantizedTensors:
        # x is of shape (B, D, ...)
        org_shape = x.shape
        x_flat = einops.rearrange(x, 'B D ... -> (B ...) D')  # -> N x D

        # Update distribution parameters
        mean, cov = self.normal_estimator(x_flat)
        C = torch.linalg.cholesky(cov)  # D x D

        standardized = (x_flat - mean) @ torch.inverse(C).T  # N x D, pull back to standard normal space
        standardized = normal_cdf(standardized)  # N x D, push to [0, 1] range using CDF

        with torch.no_grad():
            # Quantize the input
            indices_md = torch.empty_like(standardized, dtype=torch.long)  # N x D
            for d in range(self.dim):
                bins = self.bins[d, : self.num_bins[d] - 1]
                torch.bucketize(standardized[:, d].contiguous(), bins, out=indices_md[:, d], right=True)
            centers_normed = torch.take_along_dim(self.centers, indices_md.permute(1, 0), dim=1).permute(1, 0)  # N x D

        # STE
        centers_normed_ste = (
            standardized + (centers_normed - standardized).detach()
        )  # N x D, STE in standard normal space

        # Push forward to original space
        z = standardized
        z_q_ste = centers_normed_ste
        # z = standardized  @ C.T + mean  # N x D
        # z_q_ste = centers_normed_ste @ C.T + mean  # N x D

        indices = self._encode_indices(indices_md)  # N
        indices = indices.reshape(org_shape[:1] + org_shape[2:])  # B x ...

        # Reshape back to original shape, B x D x ...
        z = z.reshape(org_shape[:1] + org_shape[2:] + org_shape[1:2])
        z = einops.rearrange(z, 'B ... D -> B D ...')

        z_q_ste = z_q_ste.reshape(org_shape[:1] + org_shape[2:] + org_shape[1:2])
        z_q_ste = einops.rearrange(z_q_ste, 'B ... D -> B D ...')

        if self.kl_weight > 0:
            dims = (0,) + tuple(range(2, len(org_shape)))  # All dimensions except D
            sigma = x.var(dim=dims)
            kl_div = variance_regularizer(sigma, entropy=2.5).mean()
            self.set_loss('kl_divergence', self.kl_weight * kl_div)

        return QuantizedTensors(
            values=z_q_ste,
            indices=indices,
            pre_quantization=z,
        )
