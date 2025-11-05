from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def silu(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.silu(input._underlying))

    _infinicore.silu_(out._underlying, input._underlying)
