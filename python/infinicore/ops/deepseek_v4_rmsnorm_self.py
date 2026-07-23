from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def deepseek_v4_rmsnorm_self(input: Tensor, epsilon: float, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.deepseek_v4_rmsnorm_self(input._underlying, epsilon))
    _infinicore.deepseek_v4_rmsnorm_self_(out._underlying, input._underlying, epsilon)
    return out


def deepseek_v4_rmsnorm_self_(out: Tensor, input: Tensor, epsilon: float) -> Tensor:
    _infinicore.deepseek_v4_rmsnorm_self_(out._underlying, input._underlying, epsilon)
    return out
