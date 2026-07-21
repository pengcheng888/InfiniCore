#include "infinicore/ops/deepseek_v4_silu_and_mul_clamp.hpp"

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

namespace infinicore::op {

Tensor deepseek_v4_silu_and_mul_clamp(const Tensor &x, float swiglu_limit) {
    auto shape = x->shape();
    if (shape.empty() || shape.back() % 2 != 0) {
        throw std::runtime_error("deepseek_v4_silu_and_mul_clamp input last dim must be even.");
    }
    shape.back() /= 2;
    auto out = Tensor::empty(shape, x->dtype(), x->device());
    deepseek_v4_silu_and_mul_clamp_(out, x, swiglu_limit);
    return out;
}

void deepseek_v4_silu_and_mul_clamp_(Tensor out, const Tensor &x, float swiglu_limit) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (x->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_silu_and_mul_clamp_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (x->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_silu_and_mul_clamp_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    if (x->shape().empty() || x->shape().back() % 2 != 0) {
        throw std::runtime_error("deepseek_v4_silu_and_mul_clamp_ input last dim must be even.");
    }
    auto expected = x->shape();
    expected.back() /= 2;
    if (out->shape() != expected) {
        throw std::runtime_error("deepseek_v4_silu_and_mul_clamp_ output shape mismatch.");
    }

    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto out_at = infinicore::adaptor::to_aten_tensor(out);
    const int64_t hidden = static_cast<int64_t>(expected.back());
    const int64_t last_dim = x_at.dim() - 1;
    auto gate = x_at.narrow(last_dim, 0, hidden);
    auto up = x_at.narrow(last_dim, hidden, hidden);

    auto limit = at::full({}, static_cast<double>(swiglu_limit), gate.options());
    auto gate_clamped = at::minimum(gate, limit);
    auto up_clamped = at::clamp(up, -static_cast<double>(swiglu_limit), static_cast<double>(swiglu_limit));
    auto gate_f = gate_clamped.to(at::kFloat);
    auto up_f = up_clamped.to(at::kFloat);
    auto result = (gate_f / (1.0 + at::exp(-gate_f))) * up_f;
    out_at.copy_(result.to(out_at.scalar_type()));
#else
    (void)out;
    (void)x;
    (void)swiglu_limit;
    throw std::runtime_error("deepseek_v4_silu_and_mul_clamp_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
