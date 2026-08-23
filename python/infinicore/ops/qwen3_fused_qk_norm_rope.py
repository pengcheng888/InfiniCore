from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def qwen3_fused_qk_norm_rope_(
    qkv: Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: Tensor,
    k_weight: Tensor,
    base: float,
    is_neox: bool,
    position_ids: Tensor,
    factor: float = 1.0,
    low: float = 0.0,
    high: float = 0.0,
    attention_factor: float = 1.0,
    rotary_dim: int = 0,
) -> Tensor:
    _infinicore.qwen3_fused_qk_norm_rope_(
        qkv._underlying,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight._underlying,
        k_weight._underlying,
        base,
        is_neox,
        position_ids._underlying,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )
    return qkv


def qwen3_fused_qk_norm_rope(*args, **kwargs) -> Tensor:
    return qwen3_fused_qk_norm_rope_(*args, **kwargs)

