from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_aiter_loaded() -> None:
    import aiter  # noqa: F401


def _underlying_or_none(tensor: Optional[Tensor]):
    return None if tensor is None else tensor._underlying


def deepseek_v4_moe_marlin_w8a8_(
    input: Tensor,
    b_qweight: Tensor,
    output: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    topk_weights: Optional[Tensor],
    sorted_token_ids: Tensor,
    expert_ids: Tensor,
    num_tokens_post_pad: Tensor,
    top_k: int,
    mode: int,
    delta: int,
) -> Tensor:
    _ensure_aiter_loaded()
    _infinicore.deepseek_v4_moe_marlin_w8a8_(
        input._underlying,
        b_qweight._underlying,
        output._underlying,
        a_scale._underlying,
        b_scale._underlying,
        _underlying_or_none(topk_weights),
        sorted_token_ids._underlying,
        expert_ids._underlying,
        num_tokens_post_pad._underlying,
        top_k,
        mode,
        delta,
    )
    return output


def deepseek_v4_moe_marlin_w8a8_fp8_(
    input: Tensor,
    b_qweight: Tensor,
    output: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    topk_weights: Optional[Tensor],
    sorted_token_ids: Tensor,
    expert_ids: Tensor,
    num_tokens_post_pad: Tensor,
    top_k: int,
    mode: int,
    delta: int,
) -> Tensor:
    _ensure_aiter_loaded()
    _infinicore.deepseek_v4_moe_marlin_w8a8_fp8_(
        input._underlying,
        b_qweight._underlying,
        output._underlying,
        a_scale._underlying,
        b_scale._underlying,
        _underlying_or_none(topk_weights),
        sorted_token_ids._underlying,
        expert_ids._underlying,
        num_tokens_post_pad._underlying,
        top_k,
        mode,
        delta,
    )
    return output
