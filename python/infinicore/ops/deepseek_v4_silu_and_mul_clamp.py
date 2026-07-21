from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def deepseek_v4_silu_and_mul_clamp(input: Tensor, swiglu_limit: float, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.deepseek_v4_silu_and_mul_clamp(input._underlying, swiglu_limit))
    _infinicore.deepseek_v4_silu_and_mul_clamp_(out._underlying, input._underlying, swiglu_limit)
    return out



def deepseek_v4_silu_and_mul_clamp_(out: Tensor, input: Tensor, swiglu_limit: float) -> Tensor:
    _infinicore.deepseek_v4_silu_and_mul_clamp_(out._underlying, input._underlying, swiglu_limit)
    return out
