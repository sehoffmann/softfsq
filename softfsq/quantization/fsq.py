from typing import override, Sequence

import torch
import einops

from softfsq.quantization.common import QuantizedTensors, Quantizer

__all__ = [
    'FSQ',
]


# tensor helpers

def round_ste(z):
    """ round with straight through gradients. """
    zhat = z.round()
    return z + (zhat - z).detach()

def floor_ste(z):
    """ floor with straight through gradients. """
    zhat = z.floor()
    return z + (zhat - z).detach()


class FSQ(Quantizer):
    def __init__(self, levels: Sequence[int]):
        super().__init__()
        
        levels = torch.tensor(levels, dtype=torch.int64)
        if levels.ndim != 1:
            raise ValueError('levels must be a 1D sequence of integers.')
        self.register_buffer('levels', levels, persistent=True)

        basis = torch.cat([
            torch.tensor([1], dtype=torch.int64),
            torch.cumprod(levels[:-1], dim=0, dtype=torch.int64),
        ])  # used to convert codes (i.e. rounded values) to indices
        self.register_buffer('basis', basis, persistent=True)

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
        codes_non_centered = torch.remainder(
            torch.floor_divide(indices, self.basis), self.levels
        )
        decoded = self._scale_and_shift_inverse(codes_non_centered)
        return einops.rearrange(decoded, 'b ... c -> b c ...').contiguous()  # channels first

    @override
    def encode(self, inputs: torch.Tensor) -> QuantizedTensors:
        inputs = einops.rearrange(inputs, 'b c ... -> b ... c').contiguous()  # channels last

        with torch.autocast(device_type=inputs.device.type, enabled=False):   # run in f32!
            inputs_f32 = inputs.float()
            z_bounded = self._bound(inputs_f32) # bound to [-(L-1)/2, (L-1)/2]
            z_q = round_ste(z_bounded)

            # renormalize back to [-1, 1]
            half_width = self.levels // 2
            z_q = z_q / half_width
            z_bounded = z_bounded / half_width

            indices = self._codes_to_indexes(z_q)

        # to channels first
        z_q = einops.rearrange(z_q, 'b ... c -> b c ...').contiguous()
        z_bounded = einops.rearrange(z_bounded, 'b ... c -> b c ...').contiguous()

        return QuantizedTensors(values=z_q, indices=indices, pre_quantization=z_bounded)

if __name__ == '__main__':
    fsq = FSQ(levels=[5, 4])
    x = torch.tensor([
        [[2.0, 0.0, -2.0], [-1.0, -1.0, -1.0]],
    ])
    print(x.shape)
    print('input: ', x)
    qt = fsq.encode(x)
    print('bounded: ', qt.pre_quantization, qt.pre_quantization.shape)
    print('indices: ', qt.indices, qt.indices.shape)
    print('quant: ', qt.values, qt.values.shape)

    # decode:
    print('decoded: ', fsq.decode(qt.indices), fsq.decode(qt.indices).shape)
