import infinicore
import torch
from torch import Size, Tensor
from typing import Callable, Optional, TYPE_CHECKING, Union


__all__ = ["RMSNorm"]
_shape_t = Union[int, list[int], Size]

# class RMSNorm(infinicore.nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         LlamaRMSNorm is equivalent to LlamaRMSNorm
#         """
#         super().__init__()
#         self.weight = torch.nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps
#         self.weight_infini = None

#     def forward(self,
#                 hidden_states: Union[infinicore.Tensor, torch.Tensor]
#                 ) -> Union[infinicore.Tensor, torch.Tensor]:
#         if isinstance(hidden_states, infinicore.Tensor):
#             return self.forward_infini2infini(hidden_states)

#         return self.forward_torch2torch(hidden_states)

#     def forward_torch2torch(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         # print(" LlamaRMSNorm :: forward_torch2torch ")
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)

#     def forward_infini2infini(self, hidden_states: infinicore.Tensor) -> infinicore.Tensor:
#         from infinicore.nn.modules.linear import torch_tensor_2_infini_tensor

#         # print(" LlamaRMSNorm :: forward_infini2infini ")
#         if self.weight_infini is None:
#             self.weight_infini = torch_tensor_2_infini_tensor(self.weight)
#         return infinicore.rms_norm(hidden_states, self.weight_infini, self.variance_epsilon)

#     def extra_repr(self):
#         return f" infinicore op : {tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    

class RMSNorm(infinicore.nn.Module):
    r"""Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Root Mean Square Layer Normalization <https://arxiv.org/pdf/1910.07467.pdf>`__

    .. math::
        y_i = \frac{x_i}{\mathrm{RMS}(x)} * \gamma_i, \quad
        \text{where} \quad \text{RMS}(x) = \sqrt{\epsilon + \frac{1}{n} \sum_{i=1}^{n} x_i^2}

    The RMS is taken over the last ``D`` dimensions, where ``D``
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the RMS is computed over
    the last 2 dimensions of the input.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: :func:`torch.finfo(x.dtype).eps`
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> rms_norm = nn.RMSNorm([2, 3])
        >>> input = torch.randn(2, 2, 3)
        >>> rms_norm(input)

    """
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple[int, ...]
    eps: Optional[float]
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        raise ValueError("not support.")
