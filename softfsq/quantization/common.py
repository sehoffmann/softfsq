from abc import abstractmethod
from dataclasses import dataclass
from typing import override

import torch
from torch import nn
import dmlcloud as dml

__all__ = [
    'QuantizedTensors',
    'Quantizer',
    'IdentityQuantizer',
    'identity_quantize',
]


@dataclass
class QuantizedTensors:
    """
    Dataclass to hold quantized values and their corresponding indices.

    The returned `values` might possibly use surrogate gradients to allow backpropagation, e.g. via
    the straight-through estimator.

    If the quantizer further modifies its inputs prior to quantization (e.g., via normalization),
    these unmodified inputs are stored in the `pre_quantization` attribute.

    Attributes:
        values (torch.Tensor): Float Tensor of shape (..., D) containing quantized values.
        indices (torch.Tensor): Int64 Tensor of shape (...) containing quantized indices.
        pre_quantization (torch.Tensor, optional): Float Tensor of shape (..., D) containing unquantized values.
    """

    values: torch.Tensor
    indices: torch.Tensor
    pre_quantization: torch.Tensor = None


class Quantizer(nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized indices to continuous values.

        Args:
            indices (torch.Tensor): Int64 Tensor of shape (...) containing quantized indices.

        Returns:
            torch.Tensor: Decoded continuous values of shape (..., D).
        """
        pass

    @abstractmethod
    def encode(self, inputs: torch.Tensor) -> QuantizedTensors:
        """
        Encode continuous values to quantized indices.

        Args:
            inputs (torch.Tensor): Float Tensor of shape (..., D) containing continuous values.

        Returns:
            QuantizedTensors: A dataclass containing:
                - values (torch.Tensor): Float Tensor of shape (..., D) with quantized values.
                - indices (torch.Tensor): Int64 Tensor of shape (...) with quantized indices.
        """
        pass

    @property
    @abstractmethod
    def codebook_size(self) -> int:
        """
        Returns the size of the codebook (number of discrete codes).
        This is equivalent to the maximum index + 1.

        Returns:
            int: Size of the codebook.
        """
        pass

    @property
    @abstractmethod
    def codebook_dim(self) -> int:
        """
        Returns the dimensionality of each code in the codebook.

        Returns:
            int: Dimensionality of each code.
        """
        pass

    def __call__(self, inputs: torch.Tensor) -> QuantizedTensors:
        result = self.encode(inputs)
        residuals = result.values - result.pre_quantization
        dml.log_metric('quant_residual_l1', torch.abs(residuals).mean())
        return result


def identity_quantize(inputs: torch.Tensor) -> QuantizedTensors:
    """
    Identity quantization function that returns the inputs as is.

    Args:
        inputs (torch.Tensor): Input tensor of shape (..., D).

    Returns:
        QuantizedTensors: A dataclass containing:
            - values (torch.Tensor): Same as inputs.
            - indices (torch.Tensor): None.
            - pre_quantization (torch.Tensor): Same as inputs.
    """
    return QuantizedTensors(values=inputs, indices=None, pre_quantization=inputs)


class IdentityQuantizer(Quantizer):
    """
    Identity quantizer that performs no quantization.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    @override
    @property
    def codebook_size(self) -> int:
        return 1  # No codebook

    @override
    @property
    def codebook_dim(self) -> int:
        return self.input_dim

    @override
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('IdentityQuantizer does not support decoding.')

    @override
    def encode(self, inputs: torch.Tensor) -> QuantizedTensors:
        return QuantizedTensors(
            values=inputs, 
            indices=torch.zeros_like(inputs[:, 0]), 
            pre_quantization=inputs
        )
