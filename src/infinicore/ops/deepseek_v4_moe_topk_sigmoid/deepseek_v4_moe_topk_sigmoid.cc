#include "infinicore/ops/deepseek_v4_moe_topk_sigmoid.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/core/dispatch/Dispatcher.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <stdexcept>

namespace infinicore::op {

void deepseek_v4_moe_topk_sigmoid_(Tensor topk_weights,
                                   Tensor topk_indices,
                                   const Tensor &gating_output,
                                   bool renormalize,
                                   std::optional<Tensor> correction_bias) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (gating_output->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_moe_topk_sigmoid_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (gating_output->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_moe_topk_sigmoid_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto topk_weights_at = infinicore::adaptor::to_aten_tensor(topk_weights);
    auto topk_indices_at = infinicore::adaptor::to_aten_tensor(topk_indices);
    auto gating_output_at = infinicore::adaptor::to_aten_tensor(gating_output);
    std::optional<at::Tensor> correction_bias_at = std::nullopt;
    if (correction_bias.has_value()) {
        correction_bias_at = infinicore::adaptor::to_aten_tensor(*correction_bias);
    }

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::topk_sigmoid", "")
                         .typed<void(at::Tensor &, at::Tensor &, at::Tensor &, bool, const std::optional<at::Tensor> &)>();
    op.call(topk_weights_at, topk_indices_at, gating_output_at, renormalize, correction_bias_at);
#else
    (void)topk_weights;
    (void)topk_indices;
    (void)gating_output;
    (void)renormalize;
    (void)correction_bias;
    throw std::runtime_error("deepseek_v4_moe_topk_sigmoid_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
