from typing import override, Sequence

import dmlcloud as dml
import einops
import torch

from softfsq.math.distributions import GeneralNormalDistribution
from softfsq.quantization.common import QuantizedTensors, Quantizer

__all__ = [
    'GFSQ',
]


class GFSQ(Quantizer):
    def __init__(
        self,
        levels: Sequence[int],
        loc: float | None = None,
        scale: float | None = None,
        shape: float | None = None,
        loc_trainable=False,
        scale_trainable=True,
        shape_trainable=True,
    ):
        super().__init__()

        levels = torch.tensor(levels, dtype=torch.int64)
        if levels.ndim != 1:
            raise ValueError('levels must be a 1D sequence of integers.')
        self.register_buffer('levels', levels, persistent=False)

        basis = torch.cat(
            [
                torch.tensor([1], dtype=torch.int64),
                torch.cumprod(levels[:-1], dim=0, dtype=torch.int64),
            ]
        )  # used to convert codes (i.e. rounded values) to indices
        self.register_buffer('basis', basis, persistent=False)

        self._codebook_size = int(torch.prod(levels).item())
        self._codebook_dim = len(levels)

        self.gnd = GeneralNormalDistribution(
            loc=loc,
            scale=scale,
            shape=shape,
        )
        self.gnd.loc.requires_grad_(loc_trainable)
        self.gnd.scale_sp.requires_grad_(scale_trainable)
        self.gnd.shape_sp.requires_grad_(shape_trainable)

    @override
    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @override
    @property
    def codebook_dim(self) -> int:
        return self._codebook_dim

    def _bound(self, z):
        eps = 1e-3
        half_l = (self.levels - 1) * (1 - eps) / 2  # 1D
        offset = torch.where(self.levels % 2 == 1, 0.0, 0.5)  # 0 for even L, 1D
        shift = torch.tan(offset / half_l)  # 1D
        with torch.no_grad():
            dml.log_metric('gnd_loc', self.gnd.loc.item())
            dml.log_metric('gnd_scale', self.gnd.scale.item())
            dml.log_metric('gnd_shape', self.gnd.shape.item())
        return 2 * self.gnd.cdf(z + shift) * half_l - offset

    def _scale_and_shift(self, zhat_normalized):
        half_width = self.levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self.levels // 2
        return (zhat - half_width) / half_width

    def _codes_to_indexes(self, zhat):
        zhat = self._scale_and_shift(zhat)
        return (zhat * self.basis).sum(axis=-1).to(torch.int64)

    @override
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices[..., torch.newaxis]
        codes_non_centered = torch.remainder(torch.floor_divide(indices, self.basis), self.levels)
        decoded = self._scale_and_shift_inverse(codes_non_centered)
        return decoded

    @override
    def encode(self, inputs: torch.Tensor) -> QuantizedTensors:
        with torch.autocast(device_type=inputs.device.type, enabled=False):  # run in f64!
            inputs_f64 = inputs.double()
            z_bounded = self._bound(inputs_f64)  # bound to [-(L-1)/2, (L-1)/2]
            z_q_hard = torch.round(z_bounded)
            z_q = z_bounded + (z_q_hard - z_bounded).detach()

            # renormalize back to [-1, 1]
            half_width = self.levels // 2
            z_q = z_q / half_width
            z_q_hard = z_q_hard / half_width
            z_bounded = z_bounded / half_width

            # get indices
            indices = self._codes_to_indexes(z_q_hard)

        return QuantizedTensors(
            values=z_q, 
            indices=indices, 
            pre_quantization=z_bounded,
        )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(levels={self.levels.tolist()})'
