from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def qwen3_mha_kvcache(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    seqlens_k: Tensor,
    block_table: Tensor,
    alibi_slopes: Optional[Tensor] = None,
    scale: float = 1.0,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    if out is None:
        return Tensor(
            _infinicore.qwen3_mha_kvcache(
                q._underlying,
                k_cache._underlying,
                v_cache._underlying,
                seqlens_k._underlying,
                block_table._underlying,
                alibi_slopes._underlying if alibi_slopes is not None else None,
                scale,
            )
        )
    _infinicore.qwen3_mha_kvcache_(
        out._underlying,
        q._underlying,
        k_cache._underlying,
        v_cache._underlying,
        seqlens_k._underlying,
        block_table._underlying,
        alibi_slopes._underlying if alibi_slopes is not None else None,
        scale,
    )
    return out
