#include "infinicore/ops/deepseek_v4_moe_lmslim_marlin_w8a8.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <stdexcept>
#include <string>

namespace infinicore::op {

void deepseek_v4_moe_lmslim_marlin_w8a8_(Tensor output,
                                          const Tensor &hidden_states,
                                          const Tensor &w1,
                                          const Tensor &w2,
                                          const Tensor &topk_weights,
                                          const Tensor &topk_ids,
                                          const Tensor &w1_scale,
                                          const Tensor &w2_scale,
                                          int64_t global_num_experts,
                                          double routed_scaling_factor) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (hidden_states->device().getType() != Device::Type::HYGON || output->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_moe_lmslim_marlin_w8a8_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (hidden_states->device().getType() != Device::Type::NVIDIA || output->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_moe_lmslim_marlin_w8a8_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    c10::Stack stack;
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(hidden_states));
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(w1));
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(w2));
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(topk_weights));
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(topk_ids));
    stack.emplace_back(false);
    stack.emplace_back(std::string("silu"));
    stack.emplace_back(false);
    stack.emplace_back(false);
    stack.emplace_back(true);
    stack.emplace_back(false);
    stack.emplace_back(false);
    stack.emplace_back(true);
    stack.emplace_back(c10::SymInt(global_num_experts));
    stack.emplace_back();
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(w1_scale));
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(w2_scale));
    stack.emplace_back();
    stack.emplace_back();
    stack.emplace_back();
    stack.emplace_back();
    stack.emplace_back();
    stack.emplace_back(false);
    stack.emplace_back(routed_scaling_factor);
    stack.emplace_back();
    stack.emplace_back();
    stack.emplace_back();

    auto op = c10::Dispatcher::singleton().findSchemaOrThrow("sglang::fused_experts_impl_int8_marlin", "");
    op.callBoxed(&stack);
    auto result = stack.back().toTensor();
    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    output_at.copy_(result);
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
    throw std::runtime_error("deepseek_v4_moe_lmslim_marlin_w8a8_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
