#include "infinicore/ops/deepseek_v4/rmsnorm_self.hpp"

#include "rmsnorm_self_kernel.hpp"

#include "../../platform.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/graph/graph.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op::deepseek_v4 {

INFINICORE_GRAPH_OP_CLASS(RmsnormSelf, Tensor, const Tensor &, float);

namespace {

void check_shapes(const Tensor &out, const Tensor &x) {
    if (x->ndim() < 1 || x->size(x->ndim() - 1) == 0) {
        throw std::runtime_error("rmsnorm_self expects a non-empty last dimension.");
    }
    if (out->shape() != x->shape()) {
        throw std::runtime_error("rmsnorm_self output shape mismatch.");
    }
    if (out->dtype() != x->dtype()) {
        throw std::runtime_error("rmsnorm_self output dtype mismatch.");
    }
}

int dsv4_scalar_type_for_kernel(const Tensor &tensor, const char *op_name) {
    if (tensor->dtype() == DataType::BF16) {
        return rmsnorm_self_native::kDsv4BF16;
    }
    if (tensor->dtype() == DataType::F16) {
        return rmsnorm_self_native::kDsv4F16;
    }
    if (tensor->dtype() == DataType::F32) {
        return rmsnorm_self_native::kDsv4F32;
    }
    throw std::runtime_error(std::string(op_name) + " supports bf16/fp16/fp32 tensors only.");
}

void check_device_and_guard(const Tensor &out, const Tensor &x, const char *op_name) {
    detail::check_build_device(x, op_name);
#if defined(INFINICORE_DSV4_ACCELERATOR_API)
    if (out->device().getType() != x->device().getType() || out->device().getIndex() != x->device().getIndex()) {
        throw std::runtime_error(std::string(op_name) + " output device mismatch.");
    }
#endif
}

} // namespace

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(RmsnormSelf);

RmsnormSelf::RmsnormSelf(Tensor out, const Tensor &x, float epsilon) {
    if (out->device().getType() != x->device().getType() || out->device().getIndex() != x->device().getIndex()) {
        throw std::runtime_error("rmsnorm_self_kernel_ output device mismatch.");
    }
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, x, epsilon);
}

void RmsnormSelf::execute(Tensor out, const Tensor &x, float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(RmsnormSelf, out, x, epsilon);
}

namespace rmsnorm_self_impl {

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor x;
    int64_t rows;
    int64_t dim;
    int dtype;
    float epsilon;
};

void *plan(Tensor out, const Tensor &x, float epsilon) {
    check_device_and_guard(out, x, "rmsnorm_self_kernel_");
    check_shapes(out, x);
    if (!out->is_contiguous() || !x->is_contiguous()) {
        throw std::runtime_error("rmsnorm_self_kernel_ expects contiguous tensors.");
    }
    const int64_t dim = static_cast<int64_t>(x->size(x->ndim() - 1));
    const int64_t rows = static_cast<int64_t>(x->numel() / static_cast<size_t>(dim));
    return new PlannedMeta{graph::GraphTensor(out),
                           graph::GraphTensor(x),
                           rows,
                           dim,
                           dsv4_scalar_type_for_kernel(x, "rmsnorm_self_kernel_"),
                           epsilon};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    rmsnorm_self_native::launch_rmsnorm_self(planned->out->data(),
                                             planned->x->data(),
                                             planned->dtype,
                                             planned->rows,
                                             planned->dim,
                                             planned->epsilon,
                                             context::getStream());
#elif defined(ENABLE_ATEN) && (defined(ENABLE_METAX_API) || defined(ENABLE_ILUVATAR_API))
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    rmsnorm_self_aten_(planned->out, planned->x, planned->epsilon);
#else
    (void)planned_meta;
    throw std::runtime_error("rmsnorm_self_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace rmsnorm_self_impl

namespace rmsnorm_self_register {
INFINICORE_DSV4_GRAPH_OP_REGISTER_BUILD_DEVICE(
    RmsnormSelf,
    &rmsnorm_self_impl::plan,
    &rmsnorm_self_impl::run,
    &rmsnorm_self_impl::cleanup);
} // namespace rmsnorm_self_register

Tensor rmsnorm_self_kernel(const Tensor &x, float epsilon) {
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    rmsnorm_self_kernel_(out, x, epsilon);
    return out;
}

void rmsnorm_self_kernel_(Tensor out, const Tensor &x, float epsilon) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) \
    || (defined(ENABLE_ATEN) && (defined(ENABLE_METAX_API) || defined(ENABLE_ILUVATAR_API)))
    check_device_and_guard(out, x, "rmsnorm_self_kernel_");
    check_shapes(out, x);
    if (!out->is_contiguous() || !x->is_contiguous()) {
        throw std::runtime_error("rmsnorm_self_kernel_ expects contiguous tensors.");
    }
    RmsnormSelf::execute(out, x, epsilon);
#else
    (void)out;
    (void)x;
    (void)epsilon;
    throw std::runtime_error("rmsnorm_self_kernel_ requires an ATen-enabled HYGON/NVIDIA/METAX/ILUVATAR build.");
#endif
}

} // namespace infinicore::op::deepseek_v4
