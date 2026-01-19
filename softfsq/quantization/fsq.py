from typing import override, Sequence

import einops
import softtorch
import torch

from softfsq.quantization.common import QuantizedTensors, Quantizer

__all__ = [
    'FSQ',
]


def round_ste(z):
    """round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class FSQ(Quantizer):
    def __init__(
        self,
        levels: Sequence[int],
        mode: str = 'str',
        softness: float = 1.0,
    ):
        super().__init__()

        if mode not in ['str', 'entropic']:
            raise ValueError(f'Mode must be one of ["str", "entropic"], got {mode}')
        self.mode = mode

        if softness < 0.0:
            raise ValueError('softness must be non-negative')
        self.softness = softness

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

        scale_factor = torch.tensor(6 / 128).sqrt()
        self.register_buffer('scale_factor', scale_factor, persistent=False)

        self._codebook_size = int(torch.prod(levels).item())
        self._codebook_dim = len(levels)

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
        return torch.tanh(z + shift) * half_l - offset

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
        decoded = decoded * self.scale_factor
        return einops.rearrange(decoded, 'b ... c -> b c ...').contiguous()  # channels first

    @override
    def encode(self, inputs: torch.Tensor) -> QuantizedTensors:
        inputs = einops.rearrange(inputs, 'b c ... -> b ... c').contiguous()  # channels last

        with torch.autocast(device_type=inputs.device.type, enabled=False):  # run in f32!
            inputs_f32 = inputs.float()
            z_bounded = self._bound(inputs_f32)  # bound to [-(L-1)/2, (L-1)/2]

            if self.mode == 'str':
                z_q = round_ste(z_bounded)
            else:
                z_q = softtorch.round_st(z_bounded, mode=self.mode, softness=self.softness)

            # renormalize back to [-1, 1]
            half_width = self.levels // 2
            z_q = z_q / half_width
            z_bounded = z_bounded / half_width

            indices = self._codes_to_indexes(z_q)

            z_q = z_q * self.scale_factor
            z_bounded = z_bounded * self.scale_factor

        # to channels first
        z_q = einops.rearrange(z_q, 'b ... c -> b c ...').contiguous()
        z_bounded = einops.rearrange(z_bounded, 'b ... c -> b c ...').contiguous()

        return QuantizedTensors(values=z_q, indices=indices, pre_quantization=z_bounded)

    def __repr__(self) -> str:
        return f'FSQ(levels={self.levels.tolist()}, mode="{self.mode}", softness={self.softness})'
