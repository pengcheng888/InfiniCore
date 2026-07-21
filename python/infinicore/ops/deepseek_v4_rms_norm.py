from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def deepseek_v4_rms_norm(input: Tensor, weight: Tensor, epsilon: float, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.deepseek_v4_rms_norm(input._underlying, weight._underlying, epsilon))
    _infinicore.deepseek_v4_rms_norm_(out._underlying, input._underlying, weight._underlying, epsilon)
    return out
