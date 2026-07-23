from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_sglang_moe_op_loaded() -> None:
    import sglang.srt.layers.quantization.compressed_tensors.compressed_tensors_moe_marlin  # noqa: F401


def deepseek_v4_fused_experts_impl_int8_marlin_(
    output: Tensor,
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    w1_scale: Tensor,
    w2_scale: Tensor,
    global_num_experts: int,
    routed_scaling_factor: float = 1.0,
    inplace: bool = False,
    shared_output: Tensor | None = None,
) -> Tensor:
    _ensure_sglang_moe_op_loaded()
    _infinicore.deepseek_v4_fused_experts_impl_int8_marlin_(
        output._underlying,
        hidden_states._underlying,
        w1._underlying,
        w2._underlying,
        topk_weights._underlying,
        topk_ids._underlying,
        w1_scale._underlying,
        w2_scale._underlying,
        global_num_experts,
        routed_scaling_factor,
        inplace,
        shared_output._underlying if shared_output is not None else None,
    )
    return output


def deepseek_v4_python_fused_experts_impl_int8_marlin_(
    output: Tensor,
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    w1_scale: Tensor,
    w2_scale: Tensor,
    global_num_experts: int,
    routed_scaling_factor: float = 1.0,
    inplace: bool = False,
) -> Tensor:
    _ensure_sglang_moe_op_loaded()
    _infinicore.deepseek_v4_python_fused_experts_impl_int8_marlin_(
        output._underlying,
        hidden_states._underlying,
        w1._underlying,
        w2._underlying,
        topk_weights._underlying,
        topk_ids._underlying,
        w1_scale._underlying,
        w2_scale._underlying,
        global_num_experts,
        routed_scaling_factor,
        inplace,
    )
    return output
