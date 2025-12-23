from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def add_rms_norm(a, b, weight, epsilon=1e-5, *, out=None):
    if out is None:
        return Tensor(_infinicore.add_rms_norm(a._underlying, b._underlying, weight._underlying, epsilon))

    _infinicore.add_rms_norm_(out._underlying, a._underlying, b._underlying, weight._underlying, epsilon)

    return out
