from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_deep_gemm_loaded() -> None:
    import deepgemm  # noqa: F401


def _underlying_or_none(tensor: Optional[Tensor]):
    return None if tensor is None else tensor._underlying


def deepseek_v4_deep_gemm_low_latency_grouped_gemm_(
    matrix_a: Tensor,
    matrix_b: Tensor,
    matrix_a_scale: Tensor,
    matrix_b_scale: Tensor,
    actual_tokens: Tensor,
    matrix_c: Tensor,
    max_tokens: int,
    experts: int,
    cu_s: int,
    block_wise: bool,
    b_overlap: bool = False,
    signal: Optional[Tensor] = None,
) -> Tensor:
    _ensure_deep_gemm_loaded()
    _infinicore.deepseek_v4_deep_gemm_low_latency_grouped_gemm_(
        matrix_a._underlying,
        matrix_b._underlying,
        matrix_a_scale._underlying,
        matrix_b_scale._underlying,
        actual_tokens._underlying,
        matrix_c._underlying,
        max_tokens,
        experts,
        cu_s,
        block_wise,
        b_overlap,
        _underlying_or_none(signal),
    )
    return matrix_c


def deepseek_v4_deep_gemm_moe_w8a8_i8_marlin_prefill_down_(
    input: Tensor,
    b_qweight: Tensor,
    output: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    topk_weights: Tensor,
    sorted_token_ids: Tensor,
    expert_ids: Tensor,
    num_tokens_post_pad: Tensor,
    top_k: int,
    real_topk: int,
) -> Tensor:
    _ensure_deep_gemm_loaded()
    _infinicore.deepseek_v4_deep_gemm_moe_w8a8_i8_marlin_prefill_down_(
        input._underlying,
        b_qweight._underlying,
        output._underlying,
        a_scale._underlying,
        b_scale._underlying,
        topk_weights._underlying,
        sorted_token_ids._underlying,
        expert_ids._underlying,
        num_tokens_post_pad._underlying,
        top_k,
        real_topk,
    )
    return output


def deepseek_v4_deep_gemm_moe_w8a8_marlin_decode_down_fp8_(
    input: Tensor,
    b_qweight: Tensor,
    output: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    topk_weights: Tensor,
    sorted_token_ids: Tensor,
    expert_ids: Tensor,
    num_tokens_post_pad: Tensor,
    top_k: int,
    real_topk: int,
) -> Tensor:
    _ensure_deep_gemm_loaded()
    _infinicore.deepseek_v4_deep_gemm_moe_w8a8_marlin_decode_down_fp8_(
        input._underlying,
        b_qweight._underlying,
        output._underlying,
        a_scale._underlying,
        b_scale._underlying,
        topk_weights._underlying,
        sorted_token_ids._underlying,
        expert_ids._underlying,
        num_tokens_post_pad._underlying,
        top_k,
        real_topk,
    )
    return output
