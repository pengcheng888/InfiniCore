from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_vllm_ops_loaded() -> None:
    import vllm._C  # noqa: F401


def deepseek_v4_fused_qk_norm_rope_(
    qkv: Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: Tensor,
    k_weight: Tensor,
    cos_sin_cache: Tensor,
    is_neox: bool,
    position_ids: Tensor,
) -> Tensor:
    _ensure_vllm_ops_loaded()
    _infinicore.deepseek_v4_fused_qk_norm_rope_(
        qkv._underlying,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight._underlying,
        k_weight._underlying,
        cos_sin_cache._underlying,
        is_neox,
        position_ids._underlying,
    )
    return qkv


def deepseek_v4_fused_qk_norm_rope(*args, **kwargs) -> Tensor:
    return deepseek_v4_fused_qk_norm_rope_(*args, **kwargs)
