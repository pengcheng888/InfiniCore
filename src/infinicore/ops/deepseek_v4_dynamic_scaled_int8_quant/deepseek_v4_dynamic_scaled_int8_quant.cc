#include "infinicore/ops/deepseek_v4_dynamic_scaled_int8_quant.hpp"

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

void deepseek_v4_dynamic_scaled_int8_quant_(Tensor result,
                                            const Tensor &input,
                                            Tensor scale,
                                            std::optional<Tensor> azp) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (input->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_dynamic_scaled_int8_quant_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (input->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_dynamic_scaled_int8_quant_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto result_at = infinicore::adaptor::to_aten_tensor(result);
    auto input_at = infinicore::adaptor::to_aten_tensor(input);
    auto scale_at = infinicore::adaptor::to_aten_tensor(scale);
    std::optional<at::Tensor> azp_at = std::nullopt;
    if (azp.has_value()) {
        azp_at = infinicore::adaptor::to_aten_tensor(*azp);
    }

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("_C::dynamic_scaled_int8_quant", "")
                         .typed<void(at::Tensor &, const at::Tensor &, at::Tensor &, const std::optional<at::Tensor> &)>();
    op.call(result_at, input_at, scale_at, azp_at);
#else
    (void)result;
    (void)input;
    (void)scale;
    (void)azp;
    throw std::runtime_error("deepseek_v4_dynamic_scaled_int8_quant_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
