from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def deepseek_v4_add_rms_norm(a: Tensor, b: Tensor, weight: Tensor, epsilon: float) -> tuple[Tensor, Tensor]:
    out, residual = _infinicore.deepseek_v4_add_rms_norm(a._underlying, b._underlying, weight._underlying, epsilon)
    return Tensor(out), Tensor(residual)


def deepseek_v4_add_rms_norm_inplace(input: Tensor, residual: Tensor, weight: Tensor, epsilon: float) -> tuple[Tensor, Tensor]:
    _infinicore.deepseek_v4_add_rms_norm_inplace(input._underlying, residual._underlying, weight._underlying, epsilon)
    return input, residual
