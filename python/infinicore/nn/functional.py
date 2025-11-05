import infinicore
from infinicore.lib import _infinicore
from typing import Callable, Optional, TYPE_CHECKING, Union

__all__ = ["embedding", "rms_norm","silu","swiglu","causal_softmax" ]

def embedding(
    input: infinicore.Tensor,
    weight: infinicore.Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> infinicore.Tensor:
    r"""Generate a simple lookup table that looks up embeddings in a fixed dictionary and size.

    This module is often used to retrieve word embeddings using indices.
    The input to the module is a list of indices, and the embedding matrix,
    and the output is the corresponding word embeddings.

    See :class:`infinicore.nn.Embedding` for more details.


    .. note::
        Note that `:class:`infinicore.nn.Embedding` differs from this function in
        that it initializes the row of :attr:`weight` specified by
        :attr:`padding_idx` to all zeros on construction.

    Args:
        input (LongTensor): Tensor containing indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        padding_idx (int, optional): Unsupported parameters..
        max_norm (float, optional): Unsupported parameters..
        norm_type (float, optional): Unsupported parameters..
        sparse (bool, optional): Unsupported parameters..

    Shape:
        - Input: Long Tensor (int64) of arbitrary shape containing the indices to extract
        - Weight: Embedding matrix of floating point type with shape `(V, embedding_dim)`,
          where V = maximum index + 1 and embedding_dim = the embedding size
        - Output: `(*, embedding_dim)`, where `*` is the input shape
    """
    assert (padding_idx == None) and (max_norm == None) and (scale_grad_by_freq==False) and (sparse==False), "Unsupported parameters."

    return infinicore.Tensor(_infinicore.embedding(input._underlying, weight._underlying))



def rms_norm(
    input: infinicore.Tensor,
    normalized_shape: list[int],
    weight: infinicore.Tensor,
    eps: float = 1e-5,
) -> infinicore.Tensor:
    r"""Apply Root Mean Square Layer Normalization.

    See :class:`~infinicore.nn.RMSNorm` for details.
    """
    assert normalized_shape == weight.shape, "normalized_shape  does not match weight.shape."
    return  infinicore.Tensor(
            _infinicore.rms_norm(input._underlying, weight._underlying, eps)
        )


def silu(input:  infinicore.Tensor, inplace: bool = False) ->  infinicore.Tensor:
    r"""Apply the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    See :class:`~infinicore.nn.SiLU` for more details.
    """

    if inplace:
        return  _infinicore.silu_(input._underlying, input._underlying)
    return infinicore.Tensor(_infinicore.silu(input._underlying))




def swiglu(input, other, out=None):
    r"""Apply the Swish-Gated Linear Unit (SwiGLU) function, element-wise.
    See :class:`~infinicore.nn.SwiGLU` for more details.
    """

    if out is None:
        return infinicore.Tensor(_infinicore.swiglu(input._underlying, other._underlying))
    _infinicore.swiglu_(out._underlying, input._underlying, other._underlying)



def causal_softmax(
    input: infinicore.Tensor,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[infinicore.dtype] = None,
) -> infinicore.Tensor:
    r"""Apply a causal softmax function.

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    See :class:`~infinicore.nn.Softmax` for more details.

    Args:
        input (Tensor): input
        dim (int): Unsupported parameters..
        dtype (:class:`infinicore.dtype`, optional): Unsupported parameters.
    """
    
    assert (dim == None) and (dtype == None), "Unsupported parameters."
    return infinicore.Tensor(_infinicore.causal_softmax(input._underlying))


