from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def qwen3_silu_and_mul(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.qwen3_silu_and_mul(input._underlying))
    _infinicore.qwen3_silu_and_mul_(out._underlying, input._underlying)
    return out
