from typing import override, Sequence

import einops
import softtorch
import torch
import dmlcloud as dml

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
        mode: str = 'ste',
        softness: float = 1.0,
        softness_schedule_steps: int = 0,
    ):
        super().__init__()

        if mode not in ['ste', 'entropic']:
            raise ValueError(f'Mode must be one of ["ste", "entropic"], got {mode}')
        self.mode = mode

        if softness < 0.0:
            raise ValueError('softness must be non-negative')
        _softness = torch.tensor(softness)
        self.register_buffer('_softness', _softness, persistent=False)
        self.softness_schedule_steps = softness_schedule_steps

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

        step = torch.tensor(0, dtype=torch.int64)
        self.register_buffer('step', step, persistent=True)

        self.scaling = torch.nn.Parameter(torch.ones(len(levels)), requires_grad=True)

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

    @property
    @torch.no_grad()
    def softness(self) -> torch.Tensor:
        if self.softness_schedule_steps > 0:
            progress = self.step.float() / self.softness_schedule_steps
            softness = torch.where(
                progress <= 1.0,
                self._softness + 2 * self._softness * (1.0 - progress),
                self._softness,
            )
            return softness
        else:
            return self._softness

    def _bound(self, z):
        eps = 1e-3
        half_l = (self.levels - 1) * (1 - eps) / 2  # 1D
        offset = torch.where(self.levels % 2 == 1, 0.0, 0.5)  # 0 for even L, 1D
        shift = torch.tan(offset / half_l)  # 1D
        return torch.tanh(self.scaling * z + shift) * half_l - offset

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
        if self.training:
            self.step += 1
        softness = self.softness
        dml.log_metric('softness', softness)

        inputs = einops.rearrange(inputs, 'b c ... -> b ... c').contiguous()  # channels last
        with torch.autocast(device_type=inputs.device.type, enabled=False):  # run in f32!
            inputs_f32 = inputs.float()
            z_bounded = self._bound(inputs_f32)  # bound to [-(L-1)/2, (L-1)/2]
            z_q_hard = torch.round(z_bounded)

            if self.mode == 'ste':
                z_q = z_q_hard + (z_bounded - z_q_hard).detach()
            else:
                z_q_smooth = softtorch.round(z_bounded, mode=self.mode, softness=softness)
                if self.softness_schedule_steps:
                    z_q = torch.where(
                        self.step > self.softness_schedule_steps,
                        z_q_smooth + (z_q_hard - z_q_smooth).detach(),
                        z_q_smooth,
                    )
                else:
                    z_q = z_q_smooth + (z_q_hard - z_q_smooth).detach()

            # renormalize back to [-1, 1]
            half_width = self.levels // 2
            z_q = z_q / half_width
            z_q_hard = z_q_hard / half_width
            z_bounded = z_bounded / half_width
            
            # get indices
            indices = self._codes_to_indexes(z_q_hard)

            # normalize back to regular range
            z_q = torch.atanh(z_q * (1 - 1e-3))
            z_bounded = torch.atanh(z_bounded * (1 - 1e-3))

        # to channels first
        z_q = einops.rearrange(z_q, 'b ... c -> b c ...').contiguous()
        z_bounded = einops.rearrange(z_bounded, 'b ... c -> b c ...').contiguous()

        return QuantizedTensors(values=z_q, indices=indices, pre_quantization=z_bounded)

    def __repr__(self) -> str:
        return f'FSQ(levels={self.levels.tolist()}, mode="{self.mode}", softness={self._softness.item()}, softness_schedule_steps={self.softness_schedule_steps})'
