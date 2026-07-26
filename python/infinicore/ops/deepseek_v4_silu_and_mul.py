from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def deepseek_v4_silu_and_mul(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.deepseek_v4_silu_and_mul(input._underlying))
    _infinicore.deepseek_v4_silu_and_mul_(out._underlying, input._underlying)
    return out


def deepseek_v4_silu_and_mul_(out: Tensor, input: Tensor) -> Tensor:
    _infinicore.deepseek_v4_silu_and_mul_(out._underlying, input._underlying)
    return out
