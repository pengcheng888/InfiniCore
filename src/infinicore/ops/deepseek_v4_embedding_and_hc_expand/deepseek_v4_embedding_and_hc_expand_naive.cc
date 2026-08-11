#include "infinicore/ops/deepseek_v4_embedding_and_hc_expand.hpp"

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

namespace infinicore::op {
namespace {

Shape output_shape_for_naive(const Tensor &input, const Tensor &weight, int64_t hc_mult, const char *op_name) {
    if (weight->ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " expects weight shape [vocab, hidden].");
    }
    if (hc_mult <= 0) {
        throw std::runtime_error(std::string(op_name) + " expects hc_mult > 0.");
    }
    Shape output_shape = input->shape();
    output_shape.push_back(static_cast<size_t>(hc_mult));
    output_shape.push_back(weight->size(1));
    return output_shape;
}

void check_shapes_and_dtypes_naive(const Tensor &out, const Tensor &input, const Tensor &weight, int64_t hc_mult, const char *op_name) {
    auto expected = output_shape_for_naive(input, weight, hc_mult, op_name);
    if (out->shape() != expected) {
        throw std::runtime_error(std::string(op_name) + " output shape mismatch.");
    }
    if (out->dtype() != weight->dtype()) {
        throw std::runtime_error(std::string(op_name) + " output dtype must match weight dtype.");
    }
    if (input->dtype() != DataType::I32 && input->dtype() != DataType::I64) {
        throw std::runtime_error(std::string(op_name) + " expects int32 or int64 input indices.");
    }
}

void check_same_device_naive(const Tensor &out, const Tensor &input, const Tensor &weight, const char *op_name) {
    auto device = out->device();
    if (input->device() != device || weight->device() != device) {
        throw std::runtime_error(std::string(op_name) + " expects input, weight, and output on the same device.");
    }
}

void check_accelerator_tensor_naive(const Tensor &tensor, const char *op_name) {
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

Tensor deepseek_v4_embedding_and_hc_expand_naive(const Tensor &input, const Tensor &weight, int64_t hc_mult) {
    auto out = Tensor::empty(output_shape_for_naive(input, weight, hc_mult, "deepseek_v4_embedding_and_hc_expand_naive"), weight->dtype(), weight->device());
    deepseek_v4_embedding_and_hc_expand_naive_(out, input, weight, hc_mult);
    return out;
}

void deepseek_v4_embedding_and_hc_expand_naive_(Tensor out, const Tensor &input, const Tensor &weight, int64_t hc_mult) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor_naive(out, "deepseek_v4_embedding_and_hc_expand_naive_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    check_same_device_naive(out, input, weight, "deepseek_v4_embedding_and_hc_expand_naive_");
    check_shapes_and_dtypes_naive(out, input, weight, hc_mult, "deepseek_v4_embedding_and_hc_expand_naive_");

    auto out_at = infinicore::adaptor::to_aten_tensor(out);
    auto input_at = infinicore::adaptor::to_aten_tensor(input).reshape({-1}).to(at::kLong);
    auto weight_at = infinicore::adaptor::to_aten_tensor(weight);

    auto gathered = weight_at.index_select(0, input_at);
    auto expanded = gathered.unsqueeze(1).expand({gathered.size(0), hc_mult, gathered.size(1)}).contiguous();
    out_at.copy_(expanded.view(out_at.sizes()));
#else
    (void)out;
    (void)input;
    (void)weight;
    (void)hc_mult;
    throw std::runtime_error("deepseek_v4_embedding_and_hc_expand_naive_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
