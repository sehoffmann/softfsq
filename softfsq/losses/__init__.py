from .common import LossMixin
from .gan import DiscriminatorHingeLoss, DiscriminatorVanillaLoss, GeneratorWGANLoss
from .loss import DiscriminatorLoss, VQGANLoss
from .lpips import lpips_loss, LPIPSLoss

__all__ = [
    'LossMixin',
    'lpips_loss',
    'LPIPSLoss',
    'DiscriminatorHingeLoss',
    'DiscriminatorVanillaLoss',
    'GeneratorWGANLoss',
    'DiscriminatorLoss',
    'VQGANLoss',
]
