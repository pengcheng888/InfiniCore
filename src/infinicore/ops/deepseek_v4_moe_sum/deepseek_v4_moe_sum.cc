#include "infinicore/ops/deepseek_v4_moe_sum.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <stdexcept>

namespace infinicore::op {

void deepseek_v4_moe_sum_(Tensor output, const Tensor &input) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (input->device().getType() != Device::Type::HYGON || output->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_moe_sum_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (input->device().getType() != Device::Type::NVIDIA || output->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_moe_sum_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto input_at = infinicore::adaptor::to_aten_tensor(input);
    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    if (input_at.dim() != 3 || output_at.dim() != 2 ||
        input_at.size(0) != output_at.size(0) || input_at.size(2) != output_at.size(1)) {
        throw std::runtime_error("deepseek_v4_moe_sum_ expects input [tokens, topk, hidden] and output [tokens, hidden].");
    }
    output_at.copy_(input_at.sum(1));
#else
    (void)output;
    (void)input;
    throw std::runtime_error("deepseek_v4_moe_sum_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
