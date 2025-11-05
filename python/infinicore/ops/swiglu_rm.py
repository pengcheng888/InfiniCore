from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def swiglu(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.swiglu(input._underlying, other._underlying))

    _infinicore.swiglu_(out._underlying, input._underlying, other._underlying)
