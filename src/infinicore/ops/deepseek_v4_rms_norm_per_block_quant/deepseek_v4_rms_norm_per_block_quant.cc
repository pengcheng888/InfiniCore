#include "infinicore/ops/deepseek_v4_rms_norm_per_block_quant.hpp"

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

void deepseek_v4_rms_norm_per_block_quant_(Tensor result,
                                           const Tensor &input,
                                           const Tensor &weight,
                                           Tensor scale,
                                           float epsilon,
                                           std::optional<Tensor> scale_ub,
                                           std::optional<Tensor> residual,
                                           int group_size,
                                           bool is_scale_transposed) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (input->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_rms_norm_per_block_quant_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (input->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_rms_norm_per_block_quant_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto result_at = infinicore::adaptor::to_aten_tensor(result);
    auto input_at = infinicore::adaptor::to_aten_tensor(input);
    auto weight_at = infinicore::adaptor::to_aten_tensor(weight);
    auto scale_at = infinicore::adaptor::to_aten_tensor(scale);
    std::optional<at::Tensor> scale_ub_at = std::nullopt;
    std::optional<at::Tensor> residual_at = std::nullopt;
    if (scale_ub.has_value()) {
        scale_ub_at = infinicore::adaptor::to_aten_tensor(*scale_ub);
    }
    if (residual.has_value()) {
        residual_at = infinicore::adaptor::to_aten_tensor(*residual);
    }

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("_C::rms_norm_per_block_quant", "")
                         .typed<void(at::Tensor &, const at::Tensor &, const at::Tensor &, at::Tensor &, double, std::optional<at::Tensor>, std::optional<at::Tensor>, int64_t, bool)>();
    op.call(result_at,
            input_at,
            weight_at,
            scale_at,
            static_cast<double>(epsilon),
            scale_ub_at,
            residual_at,
            static_cast<int64_t>(group_size),
            is_scale_transposed);
#else
    (void)result;
    (void)input;
    (void)weight;
    (void)scale;
    (void)epsilon;
    (void)scale_ub;
    (void)residual;
    (void)group_size;
    (void)is_scale_transposed;
    throw std::runtime_error("deepseek_v4_rms_norm_per_block_quant_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
