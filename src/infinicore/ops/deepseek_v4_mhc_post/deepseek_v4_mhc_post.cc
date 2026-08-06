#include "infinicore/ops/deepseek_v4_mhc_post.hpp"

#include "deepseek_v4_mhc_post_kernel.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4MhcPostKernel);

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

DeepseekV4MhcPostKernel::DeepseekV4MhcPostKernel(Tensor y,
                                                 const Tensor &x,
                                                 const Tensor &residual,
                                                 const Tensor &post,
                                                 const Tensor &comb) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x, residual, post, comb);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x, residual, post, comb);
}

void DeepseekV4MhcPostKernel::execute(Tensor y,
                                      const Tensor &x,
                                      const Tensor &residual,
                                      const Tensor &post,
                                      const Tensor &comb) {
    if (mhc_graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[infinicore graph] execute DeepseekV4MhcPostKernel recording=%d\n",
                     context::isGraphRecording() ? 1 : 0);
    }
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4MhcPostKernel, y, x, residual, post, comb);
}

namespace deepseek_v4_mhc_post_graph_impl {

struct MhcPostPlannedMeta {
    graph::GraphTensor y;
    graph::GraphTensor x;
    graph::GraphTensor residual;
    graph::GraphTensor post;
    graph::GraphTensor comb;
    int64_t tokens;
    int64_t hc;
    int64_t hidden;
};

void validate_post_kernel_tensors(Tensor y,
                                  const Tensor &x,
                                  const Tensor &residual,
                                  const Tensor &post,
                                  const Tensor &comb,
                                  int64_t &tokens,
                                  int64_t &hc,
                                  int64_t &hidden,
                                  const char *op_name) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(x, op_name);
    check_dtype(x, DataType::BF16, op_name, "x");
    check_dtype(residual, DataType::BF16, op_name, "residual");
    check_dtype(y, DataType::BF16, op_name, "y");
    check_dtype(post, DataType::F32, op_name, "post");
    check_dtype(comb, DataType::F32, op_name, "comb");

    if (x->ndim() != 2 || residual->ndim() != 3 || post->ndim() != 2 || comb->ndim() != 3) {
        throw std::runtime_error(std::string(op_name) + " unexpected input rank.");
    }
    tokens = static_cast<int64_t>(residual->size(0));
    hc = static_cast<int64_t>(residual->size(1));
    hidden = static_cast<int64_t>(residual->size(2));
    if (x->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)} || y->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc), static_cast<size_t>(hidden)} || post->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc)} || comb->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc), static_cast<size_t>(hc)}) {
        throw std::runtime_error(std::string(op_name) + " shape mismatch.");
    }
    check_contiguous_tensor(y, op_name, "y");
    check_contiguous_tensor(x, op_name, "x");
    check_contiguous_tensor(residual, op_name, "residual");
    check_contiguous_tensor(post, op_name, "post");
    check_contiguous_tensor(comb, op_name, "comb");
#else
    (void)y;
    (void)x;
    (void)residual;
    (void)post;
    (void)comb;
    (void)tokens;
    (void)hc;
    (void)hidden;
    throw std::runtime_error(std::string(op_name) + " requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void *plan_post(Tensor y,
                const Tensor &x,
                const Tensor &residual,
                const Tensor &post,
                const Tensor &comb) {
    int64_t tokens = 0;
    int64_t hc = 0;
    int64_t hidden = 0;
    validate_post_kernel_tensors(y, x, residual, post, comb, tokens, hc, hidden, "deepseek_v4_mhc_post_kernel_");
    return new MhcPostPlannedMeta{
        graph::GraphTensor(y),
        graph::GraphTensor(x),
        graph::GraphTensor(residual),
        graph::GraphTensor(post),
        graph::GraphTensor(comb),
        tokens,
        hc,
        hidden};
}

void run_post(void *planned_meta) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    auto *planned = reinterpret_cast<MhcPostPlannedMeta *>(planned_meta);
    if (mhc_graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[infinicore graph] run DeepseekV4MhcPostKernel tokens=%ld hc=%ld hidden=%ld\n",
                     static_cast<long>(planned->tokens),
                     static_cast<long>(planned->hc),
                     static_cast<long>(planned->hidden));
    }
    deepseek_v4_mhc_post::launch_kernel(
        planned->y->data(),
        planned->x->data(),
        planned->residual->data(),
        reinterpret_cast<const float *>(planned->post->data()),
        reinterpret_cast<const float *>(planned->comb->data()),
        planned->tokens,
        planned->hc,
        planned->hidden,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_mhc_post_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void cleanup_post(void **planned_meta_ptr) {
    delete *reinterpret_cast<MhcPostPlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_mhc_post_graph_impl

namespace deepseek_v4_mhc_post_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4MhcPostKernel, &deepseek_v4_mhc_post_graph_impl::plan_post, &deepseek_v4_mhc_post_graph_impl::run_post, &deepseek_v4_mhc_post_graph_impl::cleanup_post);
} // namespace deepseek_v4_mhc_post_register

void deepseek_v4_mhc_post_(Tensor y,
                           const Tensor &x,
                           const Tensor &residual,
                           const Tensor &post,
                           const Tensor &comb) {
    deepseek_v4_mhc_post_kernel_(y, x, residual, post, comb);
}

void deepseek_v4_mhc_post_kernel_(Tensor y,
                                  const Tensor &x,
                                  const Tensor &residual,
                                  const Tensor &post,
                                  const Tensor &comb) {
    DeepseekV4MhcPostKernel::execute(y, x, residual, post, comb);
}

} // namespace infinicore::op
