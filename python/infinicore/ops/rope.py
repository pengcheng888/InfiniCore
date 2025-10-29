from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def rope_huai(x: Tensor,
              pos_ids: Tensor,
              sin_table: Tensor,
              cos_table: Tensor, *, out=None):
    # print("---=> ", x.shape, pos_ids.shape, sin_table.shape, cos_table.shape)

    bs, num_attention_heads, ntok, head_dim = x.shape
    x = x.permute((0, 2, 1, 3)).view((bs * ntok, num_attention_heads, head_dim))

    if out is None:
        y = Tensor(
            _infinicore.rope(x._underlying, pos_ids._underlying, sin_table._underlying, cos_table._underlying)
        )

        return y.view((bs, ntok, num_attention_heads, head_dim)).permute((0, 2, 1, 3))

    _infinicore.rope_(
        out._underlying, x._underlying, pos_ids._underlying, sin_table._underlying, cos_table._underlying
    )


def rope(x, pos_ids, sin_table, cos_table, *, out=None):
    # print("---=> ", x.shape, pos_ids.shape, sin_table.shape, cos_table.shape)

    if out is None:
        return Tensor(
            _infinicore.rope(x._underlying, pos_ids._underlying, sin_table._underlying, cos_table._underlying)
        )

    _infinicore.rope_(
        out._underlying, x._underlying, pos_ids._underlying, sin_table._underlying, cos_table._underlying
    )
