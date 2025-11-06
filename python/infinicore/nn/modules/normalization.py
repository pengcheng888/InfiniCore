import infinicore
import torch
import numbers
from typing import  Optional, Union
from .module import Module


__all__ = ["RMSNorm"]


class RMSNorm(Module):
    r"""Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    Args:
        normalized_shape (int or list): input shape from an expected input
            of size ...
    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)
    """
    __constants__ = ["normalized_shape", "eps"]
    normalized_shape: tuple[int, ...]
    eps: Optional[float]

    def __init__(
            self,
            normalized_shape: Union[int, list[int]],
            eps = 1e-6,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.weight_infini = None

    def forward(self, x: infinicore.Tensor) -> infinicore.Tensor:
        if self.weight_infini is None:
            self.weight_infini = infinicore.convert_torch_to_infini_tensor(self.weight)
        return infinicore.nn.functional.rms_norm(x, self.normalized_shape, self.weight_infini, self.eps)

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}".format(**self.__dict__)
