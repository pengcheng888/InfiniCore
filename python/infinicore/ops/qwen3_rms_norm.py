from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def qwen3_rms_norm(x: Tensor, weight: Tensor, epsilon: float = 1e-6, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.qwen3_rms_norm(x._underlying, weight._underlying, epsilon))
    _infinicore.qwen3_rms_norm_(out._underlying, x._underlying, weight._underlying, epsilon)
    return out
