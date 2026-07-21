#include "infinicore/ops/deepseek_v4_linear_bf16_fp32.hpp"

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

namespace infinicore::op {

namespace {

void check_linear_shapes(const Tensor &out, const Tensor &x, const Tensor &weight) {
    if (x->ndim() != 2 || weight->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32 expects 2D input and weight tensors.");
    }
    if (x->size(1) != weight->size(1)) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32 input/weight K dimension mismatch.");
    }
    if (out->shape() != Shape{x->size(0), weight->size(0)}) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32 output shape mismatch.");
    }
    if (out->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32 output dtype must be float32.");
    }
}

} // namespace

Tensor deepseek_v4_linear_bf16_fp32(const Tensor &x, const Tensor &weight) {
    if (x->ndim() != 2 || weight->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32 expects 2D input and weight tensors.");
    }
    if (x->size(1) != weight->size(1)) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32 input/weight K dimension mismatch.");
    }
    auto out = Tensor::empty({x->size(0), weight->size(0)}, DataType::F32, x->device());
    deepseek_v4_linear_bf16_fp32_(out, x, weight);
    return out;
}

void deepseek_v4_linear_bf16_fp32_(Tensor out, const Tensor &x, const Tensor &weight) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (x->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (x->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    check_linear_shapes(out, x, weight);
    auto out_at = infinicore::adaptor::to_aten_tensor(out);
    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto weight_at = infinicore::adaptor::to_aten_tensor(weight);
    auto result = at::matmul(x_at.to(at::kFloat), weight_at.to(at::kFloat).transpose(0, 1));
    out_at.copy_(result);
#else
    (void)out;
    (void)x;
    (void)weight;
    throw std::runtime_error("deepseek_v4_linear_bf16_fp32_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
