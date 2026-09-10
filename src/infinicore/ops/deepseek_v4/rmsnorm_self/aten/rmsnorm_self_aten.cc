#include "infinicore/ops/deepseek_v4/rmsnorm_self.hpp"

#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

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

namespace infinicore::op::deepseek_v4 {
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

void check_shapes(const Tensor &out, const Tensor &x, const char *op_name) {
    if (x->ndim() < 1) {
        throw std::runtime_error(std::string(op_name) + " expects rank >= 1.");
    }
    if (out->shape() != x->shape()) {
        throw std::runtime_error(std::string(op_name) + " output shape mismatch.");
    }
    if (out->dtype() != x->dtype()) {
        throw std::runtime_error(std::string(op_name) + " output dtype mismatch.");
    }
}

} // namespace

void rmsnorm_self_aten_(Tensor out, const Tensor &x, float epsilon) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(x, "deepseek_v4::rmsnorm_self_aten_");
    check_accelerator_tensor(out, "deepseek_v4::rmsnorm_self_aten_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    check_shapes(out, x, "deepseek_v4::rmsnorm_self_aten_");

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
    throw std::runtime_error("deepseek_v4::rmsnorm_self_aten_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op::deepseek_v4
