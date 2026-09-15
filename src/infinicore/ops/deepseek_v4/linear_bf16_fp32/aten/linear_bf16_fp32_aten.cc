#include "infinicore/ops/deepseek_v4/linear_bf16_fp32.hpp"

#include "../../aten_utils.hpp"

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

namespace infinicore::op::deepseek_v4 {
namespace {

void check_accelerator_tensor(const Tensor &tensor, const char *op_name) {
    detail::check_build_device(tensor, op_name);
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

void linear_bf16_fp32_aten_(Tensor out, const Tensor &x, const Tensor &weight) {
#if defined(ENABLE_ATEN) && defined(INFINICORE_DSV4_ACCELERATOR_API)
    check_accelerator_tensor(x, "deepseek_v4::linear_bf16_fp32_aten_");
    detail::prepare_aten_call(x, "deepseek_v4::linear_bf16_fp32_aten_");

    check_shapes(out, x, weight, "deepseek_v4::linear_bf16_fp32_aten_");
    auto out_at = infinicore::adaptor::to_aten_tensor(out);
    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto weight_at = infinicore::adaptor::to_aten_tensor(weight);
    at::mm_out(out_at, x_at, weight_at.transpose(0, 1), at::kFloat);
#else
    (void)out;
    (void)x;
    (void)weight;
    throw std::runtime_error("deepseek_v4::linear_bf16_fp32_aten_ requires an ATen-enabled HYGON/NVIDIA/METAX/ILUVATAR build.");
#endif
}

} // namespace infinicore::op::deepseek_v4
