#include "infinicore/ops/deepseek_v4_rmsnorm_self.hpp"

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

void check_shapes(const Tensor &out, const Tensor &x) {
    if (x->ndim() < 1) {
        throw std::runtime_error("deepseek_v4_rmsnorm_self_naive_ expects rank >= 1.");
    }
    if (out->shape() != x->shape()) {
        throw std::runtime_error("deepseek_v4_rmsnorm_self_naive_ output shape mismatch.");
    }
    if (out->dtype() != x->dtype()) {
        throw std::runtime_error("deepseek_v4_rmsnorm_self_naive_ output dtype mismatch.");
    }
}

void check_device_and_guard(const Tensor &out, const Tensor &x, const char *op_name) {
#if defined(ENABLE_HYGON_API)
    if (x->device().getType() != Device::Type::HYGON || out->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
#elif defined(ENABLE_NVIDIA_API)
    if (x->device().getType() != Device::Type::NVIDIA || out->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error(std::string(op_name) + " expects NVIDIA tensors in this build.");
    }
#else
    (void)out;
    (void)x;
    (void)op_name;
#endif
}

} // namespace

Tensor deepseek_v4_rmsnorm_self_naive(const Tensor &x, float epsilon) {
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    deepseek_v4_rmsnorm_self_naive_(out, x, epsilon);
    return out;
}



void deepseek_v4_rmsnorm_self_naive_(Tensor out, const Tensor &x, float epsilon) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    check_device_and_guard(out, x, "deepseek_v4_rmsnorm_self_naive_");
    check_shapes(out, x);

    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto out_at = infinicore::adaptor::to_aten_tensor(out);
    auto x_float = x_at.to(at::kFloat);
    auto variance = (x_float * x_float).mean({-1}, true);
    auto result = x_float * at::rsqrt(variance + static_cast<double>(epsilon));
    out_at.copy_(result.to(out_at.scalar_type()));
#else
    (void)out;
    (void)x;
    (void)epsilon;
    throw std::runtime_error("deepseek_v4_rmsnorm_self_naive_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}



} // namespace infinicore::op
