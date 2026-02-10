from abc import abstractmethod
from typing import override

import numpy as np
import torch
from torch import nn

from .gamma import gamma, gammainc


class MVNormalEstimator(nn.Module):
    """
    Estimate mean and covariance of a multivariate normal distribution from data
    using online bayesian inference.
    """

    def __init__(self, dim: int, lag: float | int = 3000):
        """
        Args:
            dim (int): Dimensionality of the distribution
            lag (floa | int): Lag parameter for exponential moving average update. Higher means slower updates. If int, interpreted as number of samples to lag behind.
        """
        super().__init__()

        self.dim = dim
        self.lag = lag

        self.mu_0 = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.phi = nn.Parameter(torch.eye(dim), requires_grad=False)
        self.kappa = nn.Parameter(torch.tensor(1, dtype=torch.int64), requires_grad=False)
        self.nu = nn.Parameter(torch.tensor(1, dtype=torch.int64), requires_grad=False)

    @property
    def mean(self):
        return self.mu_0

    @property
    def covariance(self):
        return self.phi / self.nu

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, self.mean.shape[0])  # Ensure x has shape (N, dim)
        N = x.shape[0]

        S = x.sum(dim=0)
        mu = S / N

        alpha = 1 - min(N * 0.9999 / self.lag, 0.9999)
        self.kappa.data = alpha * self.kappa.data
        self.nu.data = alpha * self.nu.data
        self.phi.data = self.phi.data * alpha + (1 - alpha) * torch.eye(
            self.dim, device=self.phi.device, dtype=self.phi.dtype
        )

        # mean update
        self.mu_0.data = (self.kappa.data * self.mu_0.data + S) / (self.kappa.data + N)

        # covariance update
        self.phi.data += (x - mu).T @ (x - mu)
        self.phi.data += (
            (self.kappa.data * N)
            / (self.kappa.data + N)
            * (mu - self.mu_0.data).unsqueeze(1)
            @ (mu - self.mu_0.data).unsqueeze(0)
        )

        # counter updates
        self.kappa.data = self.kappa.data + N
        self.nu.data = self.nu.data + N

    @torch.no_grad()
    def forward(self, x):
        self.update(x)
        return self.mean, self.covariance


class DifferentialDistribution(nn.Module):

    @abstractmethod
    def pdf(self, x):
        raise NotImplementedError

    @abstractmethod
    def cdf(self, x):
        raise NotImplementedError

    @abstractmethod
    def quantile(self, q):
        raise NotImplementedError


class GeneralNormalDistribution(DifferentialDistribution):

    EPS = 1e-8

    def __init__(
        self,
        loc: float | None = None,
        scale: float | None = None,
        shape: float | None = None,
    ):
        super().__init__()

        if loc is None:
            loc = 0.0
        if scale is None:
            scale = np.sqrt(2.0)  # standard normal distribution
        if shape is None:
            shape = 2.0  # standard normal distribution

        loc = torch.tensor(loc, dtype=torch.float64)
        scale = torch.tensor(scale, dtype=torch.float64)
        shape = torch.tensor(shape, dtype=torch.float64)

        """
        if loc.shape != scale.shape or loc.shape != shape.shape:
            raise ValueError('loc, scale and shape must have the same shape')

        if loc.shape != torch.Size([1]):
            raise NotImplementedError('Parameters must be scalar')
        """

        self.loc = nn.Parameter(loc)
        self.scale_sp = nn.Parameter(torch.log(torch.special.expm1(scale)))  # softplus inverse
        self.shape_sp = nn.Parameter(torch.log(torch.special.expm1(shape)))

    @property
    def scale(self):
        # we need softplus here to ensure positivity
        # unlike exp(log(scale)), softplus is numerically stable for small values of scale
        # and for log(scale) = 0 (problem with chain rule)
        return torch.nn.functional.softplus(self.scale_sp) + GeneralNormalDistribution.EPS

    @property
    def shape(self):
        return torch.nn.functional.softplus(self.shape_sp) + GeneralNormalDistribution.EPS

    @override
    @torch.autocast(device_type='cuda', enabled=False)
    def pdf(self, x: torch.Tensor):
        alpha = self.scale
        beta = self.shape
        x_f64 = x.double()
        z = torch.abs(x_f64 - self.loc) / alpha
        y = (beta / (2 * alpha * gamma(1 / beta))) * torch.exp(-(z**beta))
        return y.to(x.dtype)

    @override
    @torch.autocast(device_type='cuda', enabled=False)
    def cdf(self, x: torch.Tensor):
        alpha = self.scale
        beta = self.shape
        x_f64 = x.double()
        z = torch.abs(x_f64 - self.loc) / alpha  # torch.pow unstable if z->0, beta->1
        y = 0.5 * (1 + torch.sign(x_f64 - self.loc) * gammainc(1 / beta, (z + GeneralNormalDistribution.EPS) ** beta))
        return y.to(x.dtype)
