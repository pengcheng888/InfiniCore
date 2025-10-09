import _infinicore

from infinicore.tensor import Tensor

def zeros(size, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.zeros(size, dtype._underlying, device._underlying, pin_memory)
    )
