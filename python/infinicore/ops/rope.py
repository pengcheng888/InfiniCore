from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def rope(x, pos_ids, sin_table, cos_table, *, out=None):
    print("---=> ", x.shape, pos_ids.shape, sin_table.shape, cos_table.shape)

    if out is None:
        return Tensor(
            _infinicore.rope(x._underlying, pos_ids._underlying, sin_table._underlying, cos_table._underlying)
        )

    _infinicore.rope_(
        out._underlying, x._underlying, pos_ids._underlying, sin_table._underlying, cos_table._underlying
    )
