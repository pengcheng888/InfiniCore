#include "infinicore/ops/deepseek_v4_linear_bf16_fp32.hpp"

#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#include <ATen/ops/mm.h>
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

void check_shapes(const Tensor &out, const Tensor &x, const Tensor &weight, const char *op_name) {
    if (x->ndim() != 2 || weight->ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " expects 2D input and weight tensors.");
    }
    if (x->size(1) != weight->size(1)) {
        throw std::runtime_error(std::string(op_name) + " input/weight K dimension mismatch.");
    }
    if (out->shape() != Shape{x->size(0), weight->size(0)}) {
        throw std::runtime_error(std::string(op_name) + " output shape mismatch.");
    }
    if (out->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " output dtype must be float32.");
    }
}

} // namespace

Tensor deepseek_v4_linear_bf16_fp32_aten(const Tensor &x, const Tensor &weight) {
    if (x->ndim() != 2 || weight->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32_aten expects 2D input and weight tensors.");
    }
    if (x->size(1) != weight->size(1)) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32_aten input/weight K dimension mismatch.");
    }
    auto out = Tensor::empty({x->size(0), weight->size(0)}, DataType::F32, x->device());
    deepseek_v4_linear_bf16_fp32_aten_(out, x, weight);
    return out;
}

void deepseek_v4_linear_bf16_fp32_aten_(Tensor out, const Tensor &x, const Tensor &weight) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(x, "deepseek_v4_linear_bf16_fp32_aten_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    check_shapes(out, x, weight, "deepseek_v4_linear_bf16_fp32_aten_");
    if (x->dtype() != DataType::BF16 || weight->dtype() != DataType::BF16) {
        throw std::runtime_error("deepseek_v4_linear_bf16_fp32_aten_ expects bf16 input and weight tensors.");
    }

    auto out_at = infinicore::adaptor::to_aten_tensor(out);
    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto weight_at = infinicore::adaptor::to_aten_tensor(weight);
    at::mm_out(out_at, x_at, weight_at.transpose(0, 1), at::kFloat);
#else
    (void)out;
    (void)x;
    (void)weight;
    throw std::runtime_error("deepseek_v4_linear_bf16_fp32_aten_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
