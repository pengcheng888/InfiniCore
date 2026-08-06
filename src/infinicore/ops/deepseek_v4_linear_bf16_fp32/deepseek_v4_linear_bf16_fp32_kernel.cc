#include "infinicore/ops/deepseek_v4_linear_bf16_fp32.hpp"

#include "deepseek_v4_linear_bf16_fp32_common.hpp"
#include "deepseek_v4_linear_bf16_fp32_kernel.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4LinearBf16Fp32Kernel);

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

void check_kernel_tensors(const Tensor &out, const Tensor &x, const Tensor &weight, const char *op_name) {
    check_accelerator_tensor(x, op_name);
    deepseek_v4_linear_bf16_fp32_impl::check_shapes(out, x, weight, op_name);
    if (x->dtype() != DataType::BF16 || weight->dtype() != DataType::BF16) {
        throw std::runtime_error(std::string(op_name) + " expects bf16 input and weight tensors.");
    }
    if (!out->is_contiguous() || !x->is_contiguous() || !weight->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
}

} // namespace

DeepseekV4LinearBf16Fp32Kernel::DeepseekV4LinearBf16Fp32Kernel(Tensor out, const Tensor &x, const Tensor &weight) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, x, weight);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, x, weight);
}

void DeepseekV4LinearBf16Fp32Kernel::execute(Tensor out, const Tensor &x, const Tensor &weight) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4LinearBf16Fp32Kernel, out, x, weight);
}

namespace deepseek_v4_linear_bf16_fp32_graph_impl {

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor x;
    graph::GraphTensor weight;
    int64_t tokens;
    int64_t out_features;
    int64_t in_features;
};

void *plan(Tensor out, const Tensor &x, const Tensor &weight) {
    check_kernel_tensors(out, x, weight, "deepseek_v4_linear_bf16_fp32_kernel_");
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(x),
        graph::GraphTensor(weight),
        static_cast<int64_t>(x->size(0)),
        static_cast<int64_t>(weight->size(0)),
        static_cast<int64_t>(x->size(1))};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_linear_bf16_fp32_impl::launch_linear_bf16_fp32(
        reinterpret_cast<float *>(planned->out->data()),
        planned->x->data(),
        planned->weight->data(),
        planned->tokens,
        planned->out_features,
        planned->in_features,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_linear_bf16_fp32_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_linear_bf16_fp32_graph_impl

namespace deepseek_v4_linear_bf16_fp32_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4LinearBf16Fp32Kernel,
                                       &deepseek_v4_linear_bf16_fp32_graph_impl::plan,
                                       &deepseek_v4_linear_bf16_fp32_graph_impl::run,
                                       &deepseek_v4_linear_bf16_fp32_graph_impl::cleanup);
} // namespace deepseek_v4_linear_bf16_fp32_register

void deepseek_v4_linear_bf16_fp32_kernel_(Tensor out, const Tensor &x, const Tensor &weight) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    check_kernel_tensors(out, x, weight, "deepseek_v4_linear_bf16_fp32_kernel_");
    DeepseekV4LinearBf16Fp32Kernel::execute(out, x, weight);
#else
    (void)out;
    (void)x;
    (void)weight;
    throw std::runtime_error("deepseek_v4_linear_bf16_fp32_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
