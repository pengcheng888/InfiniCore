from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

FlashMLASchedMeta = _infinicore.FlashMLASchedMeta


def flash_mla_with_kvcache(*args, **kwargs):
    def unwrap(value):
        return value._underlying if isinstance(value, Tensor) else value

    out, lse = _infinicore.flash_mla_with_kvcache(
        *(unwrap(value) for value in args),
        **{name: unwrap(value) for name, value in kwargs.items()},
    )
    return Tensor(out), Tensor(lse)
