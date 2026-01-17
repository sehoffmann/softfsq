from .gan import DiscriminatorHingeLoss, DiscriminatorVanillaLoss, GeneratorWGANLoss
from .loss import CodebookLoss, DiscriminatorLoss, VQGANLoss
from .lpips import lpips_loss, LPIPSLoss

__all__ = [
    'lpips_loss',
    'LPIPSLoss',
    'DiscriminatorHingeLoss',
    'DiscriminatorVanillaLoss',
    'GeneratorWGANLoss',
    'CodebookLoss',
    'DiscriminatorLoss',
    'VQGANLoss',
]
