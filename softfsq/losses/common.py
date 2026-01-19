import torch
from torch import nn


class LossMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self, nn.Module):
            raise ValueError('LossMixin should be used with nn.Module subclasses')
        self._losses = {}

    @property
    def losses(self) -> dict[str, torch.Tensor]:
        return self._losses

    def set_loss(self, name: str, value: torch.Tensor):
        self._losses[name] = value

    def clear_losses(self, recursive: bool = True):
        if recursive:
            for module in self.named_modules():
                if isinstance(module, LossMixin):
                    module.clear_losses(recursive=False)
        else:
            self._losses.clear()

    @staticmethod
    def get_losses(module: nn.Module) -> dict[str, torch.Tensor]:
        losses = {}
        for name, m in module.named_modules():  # also iterates over self
            if not isinstance(m, LossMixin):
                continue
            for name, loss in m.losses.items():
                if name in losses:
                    raise ValueError(f'Duplicate loss name: {name}')
                losses[name] = loss
        return losses
