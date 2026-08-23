from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def qwen3_add_rms_norm(
    a: Tensor,
    b: Tensor,
    weight: Tensor,
    epsilon: float = 1e-6,
    *,
    out: Tensor | None = None,
    residual: Tensor | None = None,
):
    if out is None and residual is None:
        y, r = _infinicore.qwen3_add_rms_norm(a._underlying, b._underlying, weight._underlying, epsilon)
        return Tensor(y), Tensor(r)
    if out is None or residual is None:
        raise ValueError("out and residual must be provided together")
    _infinicore.qwen3_add_rms_norm_(out._underlying, residual._underlying, a._underlying, b._underlying, weight._underlying, epsilon)
    return out, residual


def qwen3_add_rms_norm_inplace(input: Tensor, residual: Tensor, weight: Tensor, epsilon: float = 1e-6):
    _infinicore.qwen3_add_rms_norm_inplace(input._underlying, residual._underlying, weight._underlying, epsilon)
    return input, residual
