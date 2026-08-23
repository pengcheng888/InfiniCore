#include "infinicore/ops/deepseek_v4_mhc_post.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
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

void check_accelerator_tensor(const Tensor &tensor, const char *op_name) {
#if defined(ENABLE_HYGON_API)
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
#elif defined(ENABLE_NVIDIA_API)
    if (tensor->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error(std::string(op_name) + " expects NVIDIA tensors in this build.");
    }
#else
    (void)tensor;
    (void)op_name;
#endif
}

} // namespace

void deepseek_v4_mhc_post_aten_(Tensor y,
                                const Tensor &x,
                                const Tensor &residual,
                                const Tensor &post,
                                const Tensor &comb) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(x, "deepseek_v4_mhc_post_aten_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto y_at = infinicore::adaptor::to_aten_tensor(y);
    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto residual_at = infinicore::adaptor::to_aten_tensor(residual);
    auto post_at = infinicore::adaptor::to_aten_tensor(post);
    auto comb_at = infinicore::adaptor::to_aten_tensor(comb);
    auto result = post_at.unsqueeze(-1) * x_at.to(at::kFloat).unsqueeze(1) + at::matmul(comb_at.transpose(1, 2), residual_at.to(at::kFloat));
    y_at.copy_(result.to(y_at.scalar_type()));
#else
    (void)y;
    (void)x;
    (void)residual;
    (void)post;
    (void)comb;
    throw std::runtime_error("deepseek_v4_mhc_post_aten_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
