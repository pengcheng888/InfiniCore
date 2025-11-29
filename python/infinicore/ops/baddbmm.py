from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def baddbmm(
    input: Tensor | None,
    batch1: Tensor,
    batch2: Tensor,
    out_dtype=None,
    *,
    beta=1,
    alpha=1,
    out: Tensor | None = None,
) -> Tensor:
    """
    Performs a batch matrix multiplication of the matrices `batch1` and `batch2`. The `input` is added to the final result.
    batch1 and batch2 must be 3-D tensors each containing the same number of matrices.

    If `batch1` is a (b, n , m) tensor, `batch2` is a (b, m , p) tensor, then `input` must be (b, n , p) tensor and `out` will be a (b, n , p) tensor.
    `alpha` and `beta` are scaling factors.

    out = beta * input + alpha * (batch1 @ batch2)

    Args:
        input (Tensor): matrix to be added
        batch1 (Tensor): the first matrix to be matrix multiplied
        batch2 (Tensor): the second matrix to be matrix multiplied
        out_dtype (dtype, optional): the dtype of the output tensor.

    Keyword args:
        beta (Number, optional): multiplier for `input`
        alpha (Number, optional): multiplier for `batch1 @ batch2`
        out (Tensor, optional): the output tensor.

    Example::
        >>> M = infinicore.empty(10, 2, 3)
        >>> batch1 = infinicore.empty(10,2, 3)
        >>> batch2 = infinicore.empty(10, 3, 3)
        >>> infinicore.baddbmm(M, batch1, batch2)
        >>> infinicore.baddbmm(None, batch1, batch2)

    """

    assert out_dtype is None, "out_dtype is not supported yet."

    if out is None:
        return Tensor(
            _infinicore.baddbmm(
                None if input is None else input._underlying,
                batch1._underlying,
                batch2._underlying,
                beta,
                alpha,
            )
        )

    _infinicore.baddbmm_(
        out._underlying,
        None if input is None else input._underlying,
        batch1._underlying,
        batch2._underlying,
        beta,
        alpha,
    )
    return out
