import _infinicore

from infinicore.tensor import Tensor

def ones(size, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.ones(size, dtype._underlying, device._underlying, pin_memory)
    )

