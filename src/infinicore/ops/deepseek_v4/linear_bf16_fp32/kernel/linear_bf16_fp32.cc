#include "infinicore/ops/deepseek_v4/linear_bf16_fp32.hpp"

#include "linear_bf16_fp32_kernel.hpp"

#include "../../platform.hpp"
#include "../../../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/graph/graph.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op::deepseek_v4 {

INFINICORE_GRAPH_OP_CLASS(LinearBf16Fp32Kernel, Tensor, const Tensor &, const Tensor &);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(LinearBf16Fp32Kernel);

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

void check_kernel_tensors(const Tensor &out, const Tensor &x, const Tensor &weight, const char *op_name) {
    check_accelerator_tensor(x, op_name);
    check_shapes(out, x, weight, op_name);
    if (x->dtype() != DataType::BF16 || weight->dtype() != DataType::BF16) {
        throw std::runtime_error(std::string(op_name) + " expects bf16 input and weight tensors.");
    }
    if (!out->is_contiguous() || !x->is_contiguous() || !weight->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
}

} // namespace

LinearBf16Fp32Kernel::LinearBf16Fp32Kernel(Tensor out, const Tensor &x, const Tensor &weight) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, x, weight);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, x, weight);
}

void LinearBf16Fp32Kernel::execute(Tensor out, const Tensor &x, const Tensor &weight) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(LinearBf16Fp32Kernel, out, x, weight);
}

namespace linear_bf16_fp32_graph_impl {

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor x;
    graph::GraphTensor weight;
    int64_t tokens;
    int64_t out_features;
    int64_t in_features;
};

void *plan(Tensor out, const Tensor &x, const Tensor &weight) {
    check_kernel_tensors(out, x, weight, "linear_bf16_fp32_kernel_");
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(x),
        graph::GraphTensor(weight),
        static_cast<int64_t>(x->size(0)),
        static_cast<int64_t>(weight->size(0)),
        static_cast<int64_t>(x->size(1))};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    linear_bf16_fp32_native::launch_linear_bf16_fp32(
        reinterpret_cast<float *>(planned->out->data()),
        planned->x->data(),
        planned->weight->data(),
        planned->tokens,
        planned->out_features,
        planned->in_features,
        context::getStream());
#elif defined(ENABLE_ATEN) && (defined(ENABLE_METAX_API) || defined(ENABLE_ILUVATAR_API))
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    linear_bf16_fp32_aten_(planned->out, planned->x, planned->weight);
#else
    (void)planned_meta;
    throw std::runtime_error("linear_bf16_fp32_kernel_ requires a HYGON/NVIDIA/METAX build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace linear_bf16_fp32_graph_impl

namespace linear_bf16_fp32_register {
INFINICORE_DSV4_GRAPH_OP_REGISTER_BUILD_DEVICE(
    LinearBf16Fp32Kernel,
    &linear_bf16_fp32_graph_impl::plan,
    &linear_bf16_fp32_graph_impl::run,
    &linear_bf16_fp32_graph_impl::cleanup);
} // namespace linear_bf16_fp32_register

void linear_bf16_fp32_kernel_(Tensor out, const Tensor &x, const Tensor &weight) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) \
    || (defined(ENABLE_ATEN) && (defined(ENABLE_METAX_API) || defined(ENABLE_ILUVATAR_API)))
    check_kernel_tensors(out, x, weight, "linear_bf16_fp32_kernel_");
    LinearBf16Fp32Kernel::execute(out, x, weight);
#else
    (void)out;
    (void)x;
    (void)weight;
    throw std::runtime_error("linear_bf16_fp32_kernel_ requires an ATen-enabled HYGON/NVIDIA/METAX/ILUVATAR build.");
#endif
}

} // namespace infinicore::op::deepseek_v4
