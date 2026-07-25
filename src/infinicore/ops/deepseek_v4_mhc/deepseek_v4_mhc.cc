#include "infinicore/ops/deepseek_v4_mhc.hpp"

#include "deepseek_v4_mhc_kernel.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
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
#include <string>
#include <cstdlib>
#include <cstdio>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4MhcPreKernel);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4MhcPostKernel);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4MhcHeadKernel);

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

DeepseekV4MhcPreKernel::DeepseekV4MhcPreKernel(Tensor y,
                                               Tensor post,
                                               Tensor comb,
                                               const Tensor &x,
                                               const Tensor &fn,
                                               const Tensor &scale,
                                               const Tensor &base,
                                               double rms_eps,
                                               double hc_eps,
                                               int sinkhorn_iters) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, post, comb, x, fn, scale, base);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(),
                                 y,
                                 post,
                                 comb,
                                 x,
                                 fn,
                                 scale,
                                 base,
                                 rms_eps,
                                 hc_eps,
                                 sinkhorn_iters);
}

void DeepseekV4MhcPreKernel::execute(Tensor y,
                                     Tensor post,
                                     Tensor comb,
                                     const Tensor &x,
                                     const Tensor &fn,
                                     const Tensor &scale,
                                     const Tensor &base,
                                     double rms_eps,
                                     double hc_eps,
                                     int sinkhorn_iters) {
    if (mhc_graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[infinicore graph] execute DeepseekV4MhcPreKernel recording=%d\n",
                     context::isGraphRecording() ? 1 : 0);
    }
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4MhcPreKernel,
                                      y,
                                      post,
                                      comb,
                                      x,
                                      fn,
                                      scale,
                                      base,
                                      rms_eps,
                                      hc_eps,
                                      sinkhorn_iters);
}

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

DeepseekV4MhcHeadKernel::DeepseekV4MhcHeadKernel(Tensor y,
                                                 const Tensor &x,
                                                 const Tensor &fn,
                                                 const Tensor &scale,
                                                 const Tensor &base,
                                                 double rms_eps,
                                                 double hc_eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x, fn, scale, base);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x, fn, scale, base, rms_eps, hc_eps);
}

void DeepseekV4MhcHeadKernel::execute(Tensor y,
                                      const Tensor &x,
                                      const Tensor &fn,
                                      const Tensor &scale,
                                      const Tensor &base,
                                      double rms_eps,
                                      double hc_eps) {
    if (mhc_graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[infinicore graph] execute DeepseekV4MhcHeadKernel recording=%d\n",
                     context::isGraphRecording() ? 1 : 0);
    }
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4MhcHeadKernel, y, x, fn, scale, base, rms_eps, hc_eps);
}

namespace deepseek_v4_mhc_graph_impl {

struct MhcPrePlannedMeta {
    graph::GraphTensor y;
    graph::GraphTensor post;
    graph::GraphTensor comb;
    graph::GraphTensor x;
    graph::GraphTensor fn;
    graph::GraphTensor scale;
    graph::GraphTensor base;
    graph::GraphTensor mixes;
    graph::GraphTensor sqsum;
    graph::GraphTensor pre;
    int64_t tokens;
    int64_t hc;
    int64_t hidden;
    double rms_eps;
    double hc_eps;
    int sinkhorn_iters;
};

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

struct MhcHeadPlannedMeta {
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

void validate_pre_kernel_tensors(Tensor y,
                                 Tensor post,
                                 Tensor comb,
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
    check_dtype(post, DataType::F32, op_name, "post");
    check_dtype(comb, DataType::F32, op_name, "comb");
    check_dtype(fn, DataType::F32, op_name, "fn");
    check_dtype(scale, DataType::F32, op_name, "scale");
    check_dtype(base, DataType::F32, op_name, "base");

    if (x->ndim() != 3 || fn->ndim() != 2 || scale->ndim() != 1 || base->ndim() != 1) {
        throw std::runtime_error(std::string(op_name) + " unexpected input rank.");
    }
    tokens = static_cast<int64_t>(x->size(0));
    hc = static_cast<int64_t>(x->size(1));
    hidden = static_cast<int64_t>(x->size(2));
    const int64_t mix_hc = (2 + hc) * hc;
    if (hc > 16) {
        throw std::runtime_error(std::string(op_name) + " supports hc <= 16.");
    }
    if (fn->shape() != Shape{static_cast<size_t>(mix_hc), static_cast<size_t>(hc * hidden)} ||
        base->size(0) != static_cast<size_t>(mix_hc) || scale->size(0) != 3 ||
        y->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)} ||
        post->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc)} ||
        comb->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc), static_cast<size_t>(hc)}) {
        throw std::runtime_error(std::string(op_name) + " shape mismatch.");
    }
    check_contiguous_tensor(y, op_name, "y");
    check_contiguous_tensor(post, op_name, "post");
    check_contiguous_tensor(comb, op_name, "comb");
    check_contiguous_tensor(x, op_name, "x");
    check_contiguous_tensor(fn, op_name, "fn");
    check_contiguous_tensor(scale, op_name, "scale");
    check_contiguous_tensor(base, op_name, "base");
#else
    (void)y;
    (void)post;
    (void)comb;
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
    if (x->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)} ||
        y->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc), static_cast<size_t>(hidden)} ||
        post->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc)} ||
        comb->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc), static_cast<size_t>(hc)}) {
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
    if (fn->shape() != Shape{static_cast<size_t>(hc), static_cast<size_t>(hc * hidden)} ||
        base->size(0) != static_cast<size_t>(hc) || scale->size(0) != 1 ||
        y->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)}) {
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

void *plan_pre(Tensor y,
               Tensor post,
               Tensor comb,
               const Tensor &x,
               const Tensor &fn,
               const Tensor &scale,
               const Tensor &base,
               double rms_eps,
               double hc_eps,
               int sinkhorn_iters) {
    int64_t tokens = 0;
    int64_t hc = 0;
    int64_t hidden = 0;
    validate_pre_kernel_tensors(y, post, comb, x, fn, scale, base, tokens, hc, hidden, "deepseek_v4_mhc_pre_kernel_");
    const int64_t mix_hc = (2 + hc) * hc;
    auto mixes = Tensor::empty({static_cast<size_t>(tokens), static_cast<size_t>(mix_hc)}, DataType::F32, x->device());
    auto sqsum = Tensor::empty({static_cast<size_t>(tokens)}, DataType::F32, x->device());
    auto pre = Tensor::empty({static_cast<size_t>(tokens), static_cast<size_t>(hc)}, DataType::F32, x->device());
    return new MhcPrePlannedMeta{
        graph::GraphTensor(y),
        graph::GraphTensor(post),
        graph::GraphTensor(comb),
        graph::GraphTensor(x),
        graph::GraphTensor(fn),
        graph::GraphTensor(scale),
        graph::GraphTensor(base),
        graph::GraphTensor(mixes),
        graph::GraphTensor(sqsum),
        graph::GraphTensor(pre),
        tokens,
        hc,
        hidden,
        rms_eps,
        hc_eps,
        sinkhorn_iters};
}

void run_pre(void *planned_meta) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    auto *planned = reinterpret_cast<MhcPrePlannedMeta *>(planned_meta);
    if (mhc_graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[infinicore graph] run DeepseekV4MhcPreKernel tokens=%ld hc=%ld hidden=%ld\n",
                     static_cast<long>(planned->tokens),
                     static_cast<long>(planned->hc),
                     static_cast<long>(planned->hidden));
    }
    deepseek_v4_mhc::launch_pre_kernel(
        planned->y->data(),
        reinterpret_cast<float *>(planned->post->data()),
        reinterpret_cast<float *>(planned->comb->data()),
        planned->x->data(),
        reinterpret_cast<const float *>(planned->fn->data()),
        reinterpret_cast<const float *>(planned->scale->data()),
        reinterpret_cast<const float *>(planned->base->data()),
        reinterpret_cast<float *>(planned->mixes->data()),
        reinterpret_cast<float *>(planned->sqsum->data()),
        reinterpret_cast<float *>(planned->pre->data()),
        planned->tokens,
        planned->hc,
        planned->hidden,
        planned->rms_eps,
        planned->hc_eps,
        planned->sinkhorn_iters,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_mhc_pre_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
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
    deepseek_v4_mhc::launch_post_kernel(
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
    validate_head_kernel_tensors(y, x, fn, scale, base, tokens, hc, hidden, "deepseek_v4_mhc_head_kernel_");
    auto mixes = Tensor::empty({static_cast<size_t>(tokens), static_cast<size_t>(hc)}, DataType::F32, x->device());
    auto sqsum = Tensor::empty({static_cast<size_t>(tokens)}, DataType::F32, x->device());
    return new MhcHeadPlannedMeta{
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
    auto *planned = reinterpret_cast<MhcHeadPlannedMeta *>(planned_meta);
    if (mhc_graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[infinicore graph] run DeepseekV4MhcHeadKernel tokens=%ld hc=%ld hidden=%ld\n",
                     static_cast<long>(planned->tokens),
                     static_cast<long>(planned->hc),
                     static_cast<long>(planned->hidden));
    }
    deepseek_v4_mhc::launch_head_kernel(
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
    throw std::runtime_error("deepseek_v4_mhc_head_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void cleanup_pre(void **planned_meta_ptr) {
    delete *reinterpret_cast<MhcPrePlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

void cleanup_post(void **planned_meta_ptr) {
    delete *reinterpret_cast<MhcPostPlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

void cleanup_head(void **planned_meta_ptr) {
    delete *reinterpret_cast<MhcHeadPlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_mhc_graph_impl

namespace deepseek_v4_mhc_pre_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4MhcPreKernel, &deepseek_v4_mhc_graph_impl::plan_pre, &deepseek_v4_mhc_graph_impl::run_pre, &deepseek_v4_mhc_graph_impl::cleanup_pre);
} // namespace deepseek_v4_mhc_pre_register

namespace deepseek_v4_mhc_post_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4MhcPostKernel, &deepseek_v4_mhc_graph_impl::plan_post, &deepseek_v4_mhc_graph_impl::run_post, &deepseek_v4_mhc_graph_impl::cleanup_post);
} // namespace deepseek_v4_mhc_post_register

namespace deepseek_v4_mhc_head_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4MhcHeadKernel, &deepseek_v4_mhc_graph_impl::plan_head, &deepseek_v4_mhc_graph_impl::run_head, &deepseek_v4_mhc_graph_impl::cleanup_head);
} // namespace deepseek_v4_mhc_head_register

void deepseek_v4_mhc_pre_kernel_(Tensor y,
                          Tensor post,
                          Tensor comb,
                          const Tensor &x,
                          const Tensor &fn,
                          const Tensor &scale,
                          const Tensor &base,
                          double rms_eps,
                          double hc_eps,
                          int sinkhorn_iters) {
    DeepseekV4MhcPreKernel::execute(y, post, comb, x, fn, scale, base, rms_eps, hc_eps, sinkhorn_iters);
}

void deepseek_v4_mhc_post_kernel_(Tensor y,
                           const Tensor &x,
                           const Tensor &residual,
                           const Tensor &post,
                           const Tensor &comb) {
    DeepseekV4MhcPostKernel::execute(y, x, residual, post, comb);
}

void deepseek_v4_mhc_head_kernel_(Tensor y,
                           const Tensor &x,
                           const Tensor &fn,
                           const Tensor &scale,
                           const Tensor &base,
                           double rms_eps,
                           double hc_eps) {
    DeepseekV4MhcHeadKernel::execute(y, x, fn, scale, base, rms_eps, hc_eps);
}

void deepseek_v4_moe_w8a8_naive_(Tensor y,
                                     const Tensor &x,
                                     const Tensor &topk_weights,
                                     const Tensor &topk_indices,
                                     const Tensor &w13,
                                     const Tensor &w13_scale,
                                     const Tensor &w2,
                                     const Tensor &w2_scale,
                                     double swiglu_limit) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(x, "deepseek_v4_moe_w8a8_naive_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    if (x->ndim() != 2 || topk_weights->ndim() != 2 || topk_indices->ndim() != 2 ||
        w13->ndim() != 3 || w13_scale->ndim() != 3 || w2->ndim() != 3 || w2_scale->ndim() != 3 ||
        y->shape() != x->shape()) {
        throw std::runtime_error("deepseek_v4_moe_w8a8_naive_ shape/rank mismatch.");
    }
    if (topk_weights->shape() != topk_indices->shape()) {
        throw std::runtime_error("deepseek_v4_moe_w8a8_naive_ topk shape mismatch.");
    }
    if (w13->size(0) != w2->size(0) || w13->size(1) != w2->size(2) * 2 ||
        w13->size(2) != x->size(1) || w2->size(1) != x->size(1)) {
        throw std::runtime_error("deepseek_v4_moe_w8a8_naive_ packed weight shape mismatch.");
    }

    auto y_at = infinicore::adaptor::to_aten_tensor(y);
    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto weights_at = infinicore::adaptor::to_aten_tensor(topk_weights).to(at::kFloat);
    auto indices_at = infinicore::adaptor::to_aten_tensor(topk_indices).to(at::kLong);
    auto w13_at = infinicore::adaptor::to_aten_tensor(w13);
    auto w13_scale_at = infinicore::adaptor::to_aten_tensor(w13_scale);
    auto w2_at = infinicore::adaptor::to_aten_tensor(w2);
    auto w2_scale_at = infinicore::adaptor::to_aten_tensor(w2_scale);

    const int64_t tokens = x_at.size(0);
    const int64_t hidden = x_at.size(1);
    const int64_t topk = indices_at.size(1);
    const int64_t num_experts = w13_at.size(0);
    const int64_t intermediate = w2_at.size(2);

    auto out = at::zeros({tokens, hidden}, x_at.options().dtype(at::kFloat));
    auto token_arange = at::arange(tokens, indices_at.options()).repeat_interleave(topk);
    auto flat_ids = indices_at.reshape({tokens * topk});
    auto flat_weights = weights_at.reshape({tokens * topk});
    auto x_float = x_at.to(at::kFloat);

    for (int64_t expert = 0; expert < num_experts; ++expert) {
        auto route_pos = at::nonzero(flat_ids == expert).flatten();
        if (route_pos.numel() == 0) {
            continue;
        }
        auto token_idx = token_arange.index_select(0, route_pos);
        auto x_e = x_float.index_select(0, token_idx);
        auto route_weight = flat_weights.index_select(0, route_pos).unsqueeze(1);

        auto w13_e = w13_at[expert].to(at::kFloat) * w13_scale_at[expert].to(at::kFloat);
        auto gate_up = at::matmul(x_e, w13_e.transpose(0, 1));
        auto gate = gate_up.slice(1, 0, intermediate);
        auto up = gate_up.slice(1, intermediate, 2 * intermediate);
        gate = at::minimum(gate, at::full({}, swiglu_limit, gate.options()));
        up = at::clamp(up, -swiglu_limit, swiglu_limit);
        auto act = (gate / (1.0 + at::exp(-gate))) * up;

        auto w2_e = w2_at[expert].to(at::kFloat) * w2_scale_at[expert].to(at::kFloat);
        auto down = at::matmul(act, w2_e.transpose(0, 1)) * route_weight;
        out.index_add_(0, token_idx, down);
    }

    y_at.copy_(out.to(y_at.scalar_type()));
#else
    (void)y;
    (void)x;
    (void)topk_weights;
    (void)topk_indices;
    (void)w13;
    (void)w13_scale;
    (void)w2;
    (void)w2_scale;
    (void)swiglu_limit;
    throw std::runtime_error("deepseek_v4_moe_w8a8_naive_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
