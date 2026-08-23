from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def qwen3_rotary_embedding_(
    positions: Tensor,
    query: Tensor,
    key: Tensor | None,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox: bool = True,
) -> tuple[Tensor, Tensor | None]:
    _infinicore.qwen3_rotary_embedding_(
        positions._underlying,
        query._underlying,
        None if key is None else key._underlying,
        head_size,
        cos_sin_cache._underlying,
        is_neox,
    )
    return query, key


def qwen3_rotary_embedding(
    positions: Tensor,
    query: Tensor,
    key: Tensor | None,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox: bool = True,
) -> tuple[Tensor, Tensor | None]:
    return qwen3_rotary_embedding_(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
    )

