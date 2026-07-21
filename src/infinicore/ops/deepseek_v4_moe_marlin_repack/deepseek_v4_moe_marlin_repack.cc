#include "infinicore/ops/deepseek_v4_moe_marlin_repack.hpp"

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

void deepseek_v4_moe_marlin_repack_(Tensor output, const Tensor &weight) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (weight->device().getType() != Device::Type::HYGON || output->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_moe_marlin_repack_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (weight->device().getType() != Device::Type::NVIDIA || output->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_moe_marlin_repack_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto weight_at = infinicore::adaptor::to_aten_tensor(weight);
    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    if (weight_at.dim() != 3 || weight_at.size(0) != output_at.size(0)) {
        throw std::runtime_error("deepseek_v4_moe_marlin_repack_ expects weight [E,N,K] and matching expert count.");
    }
    if (output_at.dim() != 3 && output_at.dim() != 7) {
        throw std::runtime_error("deepseek_v4_moe_marlin_repack_ expects output [E,K/64,N*64] for GEMM1 or [E,K/64,N/16,1,4,16,16] for GEMM2.");
    }
    for (int64_t expert = 0; expert < weight_at.size(0); ++expert) {
        auto src = weight_at.select(0, expert);
        auto dst = output_at.select(0, expert);
        auto transposed = src.transpose(0, 1).contiguous();
        if (output_at.dim() == 3) {
            const int64_t size_k = transposed.size(0);
            const int64_t size_n = transposed.size(1);
            if (size_k % 64 != 0) {
                throw std::runtime_error("deepseek_v4_moe_marlin_repack_ layout requires K divisible by 64.");
            }
            auto tmp = transposed
                           .reshape({size_k / 64, 64, size_n})
                           .transpose(1, 2)
                           .contiguous();
            dst.copy_(tmp.view(dst.sizes()));
        } else {
            const int64_t size_k = transposed.size(0);
            const int64_t size_n = transposed.size(1);
            if (size_k % 64 != 0 || size_n % 16 != 0) {
                throw std::runtime_error("deepseek_v4_moe_marlin_repack_ GEMM2 layout requires K divisible by 64 and N divisible by 16.");
            }
            auto tmp = transposed
                           .reshape({size_k / 64, 64, size_n / 16, 16})
                           .permute({0, 2, 3, 1})
                           .contiguous()
                           .view({size_k / 64, size_n / 16, 1, 16, 4, 16})
                           .permute({0, 1, 2, 4, 3, 5})
                           .contiguous();
            dst.copy_(tmp.view(dst.sizes()));
        }
    }
#else
    (void)output;
    (void)weight;
    throw std::runtime_error("deepseek_v4_moe_marlin_repack_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
