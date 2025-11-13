from typing import Optional

import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

__all__ = ["causal_softmax", "rms_norm", "silu", "swiglu"]


def causal_softmax(input: Tensor, out=None) -> Tensor:
    r"""Apply a causal softmax function."""

    if out is None:
        return Tensor(_infinicore.causal_softmax(input._underlying))

    _infinicore.causal_softmax_(out._underlying, input._underlying)

    return out


def rms_norm(
    input: Tensor,
    normalized_shape: list[int],
    weight: Tensor,
    eps: float = 1e-5,
    *,
    out=None,
) -> Tensor:
    r"""Apply Root Mean Square Layer Normalization."""

    assert normalized_shape == weight.shape, (
        "normalized_shape does not match weight.shape."
    )

    if out is None:
        return Tensor(_infinicore.rms_norm(input._underlying, weight._underlying, eps))

    _infinicore.rms_norm_(out._underlying, input._underlying, weight._underlying, eps)

    return out


def silu(input: Tensor, inplace: bool = False, *, out=None) -> Tensor:
    r"""Apply the Sigmoid Linear Unit (SiLU) function, element-wise."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.silu(input, inplace=inplace)

    if inplace:
        _infinicore.silu_(input._underlying, input._underlying)
        return input

    if out is None:
        return Tensor(_infinicore.silu(input._underlying))

    _infinicore.silu_(out._underlying, input._underlying)

    return out


def swiglu(input: Tensor, other: Tensor, *, out=None):
    r"""Apply the Swish-Gated Linear Unit (SwiGLU) function, element-wise."""

    if out is None:
        return Tensor(_infinicore.swiglu(input._underlying, other._underlying))

    _infinicore.swiglu_(out._underlying, input._underlying, other._underlying)

    return out


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    *,
    out=None,
) -> Tensor:
    r"""Computes scaled dot product attention on query, key and value tensors."""
    assert (attn_mask is None) and (0.0 == dropout_p), "Unsupported parameters."
    assert (enable_gqa is True) and (is_causal is True), "Incorrect parameter value."

    ntoken = query.shape[-2]
    total_token = key.shape[-2]

    assert (1 == ntoken and total_token > 1) or (ntoken == total_token), (
        "Incorrect parameter value."
    )

    if out is None:
        return infinicore.Tensor(
            _infinicore.scaled_dot_product_attention(
                query._underlying, key._underlying, value._underlying, scale
            )
        )

    _infinicore.scaled_dot_product_attention_(
        out._underlying,
        query._underlying,
        key._underlying,
        value._underlying,
        scale,
    )

    return out
