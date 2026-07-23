#include "infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <optional>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace {

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
using SglangInt8MarlinSchema = at::Tensor(
    const at::Tensor &,       // hidden_states
    const at::Tensor &,       // w1
    const at::Tensor &,       // w2
    const at::Tensor &,       // topk_weights
    const at::Tensor &,       // topk_ids
    bool,                     // inplace
    c10::string_view,         // activation
    bool,                     // apply_router_weight_on_input
    bool,                     // use_fp8_w8a8
    bool,                     // use_int8_w8a8
    bool,                     // use_int8_w8a16
    bool,                     // use_int4_w4a16
    bool,                     // per_channel_quant
    c10::SymInt,              // global_num_experts
    const std::optional<at::Tensor> &, // expert_map
    const std::optional<at::Tensor> &, // w1_scale
    const std::optional<at::Tensor> &, // w2_scale
    const std::optional<at::Tensor> &, // w1_zp
    const std::optional<at::Tensor> &, // w2_zp
    const std::optional<at::Tensor> &, // a1_scale
    const std::optional<at::Tensor> &, // a2_scale
    at::OptionalSymIntArrayRef,        // block_shape
    std::optional<bool>,               // use_nn_moe
    std::optional<double>,             // routed_scaling_factor
    const std::optional<at::Tensor> &, // shared_output
    const std::optional<at::Tensor> &, // i_q
    const std::optional<at::Tensor> &  // i_s
);

const c10::TypedOperatorHandle<SglangInt8MarlinSchema> &sglang_int8_marlin_op() {
    static const auto op = c10::Dispatcher::singleton()
                               .findSchemaOrThrow("sglang::fused_experts_impl_int8_marlin", "")
                               .typed<SglangInt8MarlinSchema>();
    return op;
}
#endif

} // namespace

void deepseek_v4_python_fused_experts_impl_int8_marlin_(Tensor output,
                                                        const Tensor &hidden_states,
                                                        const Tensor &w1,
                                                        const Tensor &w2,
                                                        const Tensor &topk_weights,
                                                        const Tensor &topk_ids,
                                                        const Tensor &w1_scale,
                                                        const Tensor &w2_scale,
                                                        int64_t global_num_experts,
                                                        double routed_scaling_factor,
                                                        bool inplace) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    // Compatibility and diagnostics path only.
    //
    // This intentionally preserves the previous integration style that enters
    // SGLang's torch.library operator. Depending on how the SGLang op is
    // registered, the call can cross the Python dispatcher and contend on the
    // Python GIL. Keep InfiniLM routed expert hot paths on
    // deepseek_v4_fused_experts_impl_int8_marlin_ instead.
#if defined(ENABLE_HYGON_API)
    if (hidden_states->device().getType() != Device::Type::HYGON || output->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_python_fused_experts_impl_int8_marlin_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (hidden_states->device().getType() != Device::Type::NVIDIA || output->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_python_fused_experts_impl_int8_marlin_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto hidden_states_at = infinicore::adaptor::to_aten_tensor(hidden_states);
    auto w1_at = infinicore::adaptor::to_aten_tensor(w1);
    auto w2_at = infinicore::adaptor::to_aten_tensor(w2);
    auto topk_weights_at = infinicore::adaptor::to_aten_tensor(topk_weights);
    auto topk_ids_at = infinicore::adaptor::to_aten_tensor(topk_ids);
    auto w1_scale_at = infinicore::adaptor::to_aten_tensor(w1_scale);
    auto w2_scale_at = infinicore::adaptor::to_aten_tensor(w2_scale);

    const std::optional<at::Tensor> none_tensor = std::nullopt;
    const std::optional<at::Tensor> w1_scale_opt = w1_scale_at;
    const std::optional<at::Tensor> w2_scale_opt = w2_scale_at;
    const at::OptionalSymIntArrayRef block_shape = std::nullopt;
    const std::optional<bool> use_nn_moe = false;
    const std::optional<double> routed_scale = routed_scaling_factor;

    auto result = sglang_int8_marlin_op().call(
        hidden_states_at,
        w1_at,
        w2_at,
        topk_weights_at,
        topk_ids_at,
        inplace,
        c10::string_view("silu"),
        false,
        false,
        true,
        false,
        false,
        true,
        c10::SymInt(global_num_experts),
        none_tensor,
        w1_scale_opt,
        w2_scale_opt,
        none_tensor,
        none_tensor,
        none_tensor,
        none_tensor,
        block_shape,
        use_nn_moe,
        routed_scale,
        none_tensor,
        none_tensor,
        none_tensor);

    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    if (output_at.data_ptr() != result.data_ptr()) {
        output_at.copy_(result);
    }
#else
    (void)output;
    (void)hidden_states;
    (void)w1;
    (void)w2;
    (void)topk_weights;
    (void)topk_ids;
    (void)w1_scale;
    (void)w2_scale;
    (void)global_num_experts;
    (void)routed_scaling_factor;
    (void)inplace;
    throw std::runtime_error("deepseek_v4_python_fused_experts_impl_int8_marlin_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
