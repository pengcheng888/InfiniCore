#include "infinicore/ops/deepseek_v4_hc_head.hpp"

#include "deepseek_v4_hc_head_kernel.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4HcHeadKernel);

namespace {

bool mhc_graph_debug_enabled() {
    static const bool enabled = []() {
        const char *value = std::getenv("INFINICORE_GRAPH_DEBUG");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

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
void check_dtype(const Tensor &tensor, DataType dtype, const char *op_name, const char *arg_name) {
    if (tensor->dtype() != dtype) {
        throw std::runtime_error(std::string(op_name) + " unexpected dtype for " + arg_name + ": expected " + toString(dtype) + ", got " + toString(tensor->dtype()));
    }
}

void check_contiguous_tensor(const Tensor &tensor, const char *op_name, const char *arg_name) {
    if (!tensor->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensor: " + arg_name);
    }
}

#endif

} // namespace

DeepseekV4HcHeadKernel::DeepseekV4HcHeadKernel(Tensor y,
                                                 const Tensor &x,
                                                 const Tensor &fn,
                                                 const Tensor &scale,
                                                 const Tensor &base,
                                                 double rms_eps,
                                                 double hc_eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x, fn, scale, base);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x, fn, scale, base, rms_eps, hc_eps);
}

void DeepseekV4HcHeadKernel::execute(Tensor y,
                                      const Tensor &x,
                                      const Tensor &fn,
                                      const Tensor &scale,
                                      const Tensor &base,
                                      double rms_eps,
                                      double hc_eps) {
    if (mhc_graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[infinicore graph] execute DeepseekV4HcHeadKernel recording=%d\n",
                     context::isGraphRecording() ? 1 : 0);
    }
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4HcHeadKernel, y, x, fn, scale, base, rms_eps, hc_eps);
}

namespace deepseek_v4_hc_head_graph_impl {

struct HcHeadPlannedMeta {
    graph::GraphTensor y;
    graph::GraphTensor x;
    graph::GraphTensor fn;
    graph::GraphTensor scale;
    graph::GraphTensor base;
    graph::GraphTensor mixes;
    graph::GraphTensor sqsum;
    int64_t tokens;
    int64_t hc;
    int64_t hidden;
    double rms_eps;
    double hc_eps;
};

void validate_head_kernel_tensors(Tensor y,
                                  const Tensor &x,
                                  const Tensor &fn,
                                  const Tensor &scale,
                                  const Tensor &base,
                                  int64_t &tokens,
                                  int64_t &hc,
                                  int64_t &hidden,
                                  const char *op_name) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(x, op_name);
    check_dtype(x, DataType::BF16, op_name, "x");
    check_dtype(y, DataType::BF16, op_name, "y");
    check_dtype(fn, DataType::F32, op_name, "fn");
    check_dtype(scale, DataType::F32, op_name, "scale");
    check_dtype(base, DataType::F32, op_name, "base");

    if (x->ndim() != 3 || fn->ndim() != 2 || scale->ndim() != 1 || base->ndim() != 1) {
        throw std::runtime_error(std::string(op_name) + " unexpected input rank.");
    }
    tokens = static_cast<int64_t>(x->size(0));
    hc = static_cast<int64_t>(x->size(1));
    hidden = static_cast<int64_t>(x->size(2));
    if (fn->shape() != Shape{static_cast<size_t>(hc), static_cast<size_t>(hc * hidden)} || base->size(0) != static_cast<size_t>(hc) || scale->size(0) != 1 || y->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)}) {
        throw std::runtime_error(std::string(op_name) + " shape mismatch.");
    }
    check_contiguous_tensor(y, op_name, "y");
    check_contiguous_tensor(x, op_name, "x");
    check_contiguous_tensor(fn, op_name, "fn");
    check_contiguous_tensor(scale, op_name, "scale");
    check_contiguous_tensor(base, op_name, "base");
#else
    (void)y;
    (void)x;
    (void)fn;
    (void)scale;
    (void)base;
    (void)tokens;
    (void)hc;
    (void)hidden;
    throw std::runtime_error(std::string(op_name) + " requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void *plan_head(Tensor y,
                const Tensor &x,
                const Tensor &fn,
                const Tensor &scale,
                const Tensor &base,
                double rms_eps,
                double hc_eps) {
    int64_t tokens = 0;
    int64_t hc = 0;
    int64_t hidden = 0;
    validate_head_kernel_tensors(y, x, fn, scale, base, tokens, hc, hidden, "deepseek_v4_hc_head_kernel_");
    auto mixes = Tensor::empty({static_cast<size_t>(tokens), static_cast<size_t>(hc)}, DataType::F32, x->device());
    auto sqsum = Tensor::empty({static_cast<size_t>(tokens)}, DataType::F32, x->device());
    return new HcHeadPlannedMeta{
        graph::GraphTensor(y),
        graph::GraphTensor(x),
        graph::GraphTensor(fn),
        graph::GraphTensor(scale),
        graph::GraphTensor(base),
        graph::GraphTensor(mixes),
        graph::GraphTensor(sqsum),
        tokens,
        hc,
        hidden,
        rms_eps,
        hc_eps};
}

void run_head(void *planned_meta) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    auto *planned = reinterpret_cast<HcHeadPlannedMeta *>(planned_meta);
    if (mhc_graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[infinicore graph] run DeepseekV4HcHeadKernel tokens=%ld hc=%ld hidden=%ld\n",
                     static_cast<long>(planned->tokens),
                     static_cast<long>(planned->hc),
                     static_cast<long>(planned->hidden));
    }
    deepseek_v4_hc_head::launch_kernel(
        planned->y->data(),
        planned->x->data(),
        reinterpret_cast<const float *>(planned->fn->data()),
        reinterpret_cast<const float *>(planned->scale->data()),
        reinterpret_cast<const float *>(planned->base->data()),
        reinterpret_cast<float *>(planned->mixes->data()),
        reinterpret_cast<float *>(planned->sqsum->data()),
        planned->tokens,
        planned->hc,
        planned->hidden,
        planned->rms_eps,
        planned->hc_eps,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_hc_head_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void cleanup_head(void **planned_meta_ptr) {
    delete *reinterpret_cast<HcHeadPlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_hc_head_graph_impl

namespace deepseek_v4_hc_head_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4HcHeadKernel, &deepseek_v4_hc_head_graph_impl::plan_head, &deepseek_v4_hc_head_graph_impl::run_head, &deepseek_v4_hc_head_graph_impl::cleanup_head);
} // namespace deepseek_v4_hc_head_register

void deepseek_v4_hc_head_(Tensor y,
                           const Tensor &x,
                           const Tensor &fn,
                           const Tensor &scale,
                           const Tensor &base,
                           double rms_eps,
                           double hc_eps) {
    deepseek_v4_hc_head_kernel_(y, x, fn, scale, base, rms_eps, hc_eps);
}

void deepseek_v4_hc_head_kernel_(Tensor y,
                                  const Tensor &x,
                                  const Tensor &fn,
                                  const Tensor &scale,
                                  const Tensor &base,
                                  double rms_eps,
                                  double hc_eps) {
    DeepseekV4HcHeadKernel::execute(y, x, fn, scale, base, rms_eps, hc_eps);
}

} // namespace infinicore::op
