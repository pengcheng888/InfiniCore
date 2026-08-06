#include "infinicore/ops/deepseek_v4_hc_head.hpp"

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

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
at::Tensor unweighted_rmsnorm(const at::Tensor &x, double eps) {
    return x * at::rsqrt(x.square().mean(-1, true) + eps);
}
#endif

} // namespace

void deepseek_v4_hc_head_naive_(Tensor y,
                                 const Tensor &x,
                                 const Tensor &fn,
                                 const Tensor &scale,
                                 const Tensor &base,
                                 double rms_eps,
                                 double hc_eps) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(x, "deepseek_v4_hc_head_naive_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    if (x->ndim() != 3 || fn->ndim() != 2 || scale->ndim() != 1 || base->ndim() != 1) {
        throw std::runtime_error("deepseek_v4_hc_head_naive_ unexpected input rank.");
    }
    const int64_t tokens = static_cast<int64_t>(x->size(0));
    const int64_t hc = static_cast<int64_t>(x->size(1));
    const int64_t hidden = static_cast<int64_t>(x->size(2));
    if (fn->shape() != Shape{static_cast<size_t>(hc), static_cast<size_t>(hc * hidden)} || base->size(0) != static_cast<size_t>(hc) || scale->size(0) != 1 || y->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)}) {
        throw std::runtime_error("deepseek_v4_hc_head_naive_ shape mismatch.");
    }

    auto y_at = infinicore::adaptor::to_aten_tensor(y);
    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto fn_at = infinicore::adaptor::to_aten_tensor(fn).to(at::kFloat);
    auto scale_at = infinicore::adaptor::to_aten_tensor(scale).to(at::kFloat);
    auto base_at = infinicore::adaptor::to_aten_tensor(base).to(at::kFloat);
    auto x_flat = x_at.reshape({tokens, hc * hidden}).to(at::kFloat);
    auto flat = unweighted_rmsnorm(x_flat, rms_eps);
    auto mixes = at::matmul(flat, fn_at.transpose(0, 1));
    auto pre = at::sigmoid(mixes * scale_at[0] + base_at) + hc_eps;
    auto result = (pre.unsqueeze(-1) * x_at.to(at::kFloat)).sum(1);
    y_at.copy_(result.to(y_at.scalar_type()));
#else
    (void)y;
    (void)x;
    (void)fn;
    (void)scale;
    (void)base;
    (void)rms_eps;
    (void)hc_eps;
    throw std::runtime_error("deepseek_v4_hc_head_naive_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
