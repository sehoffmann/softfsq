import torch
from torch.amp import custom_bwd, custom_fwd

__all__ = ['gammainc', 'gamma', 'lower_incomplete_gamma', 'upper_incomplete_gamma']


class CustomIGamma(torch.autograd.Function):
    H = 1e-7
    EPS = 1e-10

    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float64)
    def forward(ctx, a: torch.Tensor, z: torch.Tensor):  # type: ignore # pylint: disable=W0221
        a = a.double()
        z = z.double()
        a_eps = a + CustomIGamma.EPS
        z_eps = z + CustomIGamma.EPS
        y = torch.igamma(a_eps, z_eps)
        ctx.save_for_backward(a_eps, z_eps)
        return y

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):  # pylint: disable=W0221
        a, z = ctx.saved_tensors
        # we MUST! compute forward differences here, because torch.igamma returns NaN for a<0 or z<0
        d_igamma_a = (torch.igamma(a + CustomIGamma.H, z) - torch.igamma(a, z)) / CustomIGamma.H
        d_igamma_z = (torch.igamma(a, z + CustomIGamma.H) - torch.igamma(a, z)) / CustomIGamma.H
        return (grad_output * d_igamma_a, grad_output * d_igamma_z)


def gammainc(s, x):
    """
    Regularized incomplete gamma function, i.e. the CDF of the gamma distribution with shape s and scale 1.
    """
    return CustomIGamma.apply(s, x)


def gamma(x):
    """
    Gamma function
    """
    return torch.exp(torch.lgamma(x))


def lower_incomplete_gamma(s, x):
    """
    Lower incomplete gamma function
    """
    return CustomIGamma.apply(s, x) * gamma(s)


def upper_incomplete_gamma(s, x):
    """
    Upper incomplete gamma function
    """
    return (1 - CustomIGamma.apply(s, x)) * gamma(s)
