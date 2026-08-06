#include "infinicore/ops/deepseek_v4_mhc_pre.hpp"

#include "deepseek_v4_mhc_pre_kernel.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4MhcPreKernel);

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
                                               const Tensor &residual,
                                               const Tensor &fn,
                                               const Tensor &hc_scale,
                                               const Tensor &hc_base,
                                               double rms_eps,
                                               double hc_pre_eps,
                                               double hc_sinkhorn_eps,
                                               int sinkhorn_repeat) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, post, comb, residual, fn, hc_scale, hc_base);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(),
                                 y,
                                 post,
                                 comb,
                                 residual,
                                 fn,
                                 hc_scale,
                                 hc_base,
                                 rms_eps,
                                 hc_pre_eps,
                                 hc_sinkhorn_eps,
                                 sinkhorn_repeat);
}

void DeepseekV4MhcPreKernel::execute(Tensor y,
                                     Tensor post,
                                     Tensor comb,
                                     const Tensor &residual,
                                     const Tensor &fn,
                                     const Tensor &hc_scale,
                                     const Tensor &hc_base,
                                     double rms_eps,
                                     double hc_pre_eps,
                                     double hc_sinkhorn_eps,
                                     int sinkhorn_repeat) {
    if (mhc_graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[infinicore graph] execute DeepseekV4MhcPreKernel recording=%d\n",
                     context::isGraphRecording() ? 1 : 0);
    }
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4MhcPreKernel,
                                      y,
                                      post,
                                      comb,
                                      residual,
                                      fn,
                                      hc_scale,
                                      hc_base,
                                      rms_eps,
                                      hc_pre_eps,
                                      hc_sinkhorn_eps,
                                      sinkhorn_repeat);
}

namespace deepseek_v4_mhc_pre_graph_impl {

struct MhcPrePlannedMeta {
    graph::GraphTensor y;
    graph::GraphTensor post;
    graph::GraphTensor comb;
    graph::GraphTensor residual;
    graph::GraphTensor fn;
    graph::GraphTensor hc_scale;
    graph::GraphTensor hc_base;
    graph::GraphTensor mixes;
    graph::GraphTensor sqsum;
    graph::GraphTensor pre;
    int64_t tokens;
    int64_t hc;
    int64_t hidden;
    double rms_eps;
    double hc_pre_eps;
    double hc_sinkhorn_eps;
    int sinkhorn_repeat;
};

void validate_pre_kernel_tensors(Tensor y,
                                 Tensor post,
                                 Tensor comb,
                                 const Tensor &residual,
                                 const Tensor &fn,
                                 const Tensor &hc_scale,
                                 const Tensor &hc_base,
                                 int64_t &tokens,
                                 int64_t &hc,
                                 int64_t &hidden,
                                 const char *op_name) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(residual, op_name);
    check_dtype(residual, DataType::BF16, op_name, "residual");
    check_dtype(y, DataType::BF16, op_name, "y");
    check_dtype(post, DataType::F32, op_name, "post");
    check_dtype(comb, DataType::F32, op_name, "comb");
    check_dtype(fn, DataType::F32, op_name, "fn");
    check_dtype(hc_scale, DataType::F32, op_name, "hc_scale");
    check_dtype(hc_base, DataType::F32, op_name, "hc_base");

    if (residual->ndim() != 3 || fn->ndim() != 2 || hc_scale->ndim() != 1 || hc_base->ndim() != 1) {
        throw std::runtime_error(std::string(op_name) + " unexpected input rank.");
    }
    tokens = static_cast<int64_t>(residual->size(0));
    hc = static_cast<int64_t>(residual->size(1));
    hidden = static_cast<int64_t>(residual->size(2));
    const int64_t mix_hc = (2 + hc) * hc;
    if (hc > 16) {
        throw std::runtime_error(std::string(op_name) + " supports hc <= 16.");
    }
    if (fn->shape() != Shape{static_cast<size_t>(mix_hc), static_cast<size_t>(hc * hidden)} || hc_base->size(0) != static_cast<size_t>(mix_hc) || hc_scale->size(0) != 3 || y->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)} || post->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc)} || comb->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc), static_cast<size_t>(hc)}) {
        throw std::runtime_error(std::string(op_name) + " shape mismatch.");
    }
    check_contiguous_tensor(y, op_name, "y");
    check_contiguous_tensor(post, op_name, "post");
    check_contiguous_tensor(comb, op_name, "comb");
    check_contiguous_tensor(residual, op_name, "residual");
    check_contiguous_tensor(fn, op_name, "fn");
    check_contiguous_tensor(hc_scale, op_name, "hc_scale");
    check_contiguous_tensor(hc_base, op_name, "hc_base");
#else
    (void)y;
    (void)post;
    (void)comb;
    (void)residual;
    (void)fn;
    (void)hc_scale;
    (void)hc_base;
    (void)tokens;
    (void)hc;
    (void)hidden;
    throw std::runtime_error(std::string(op_name) + " requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

/*
plan_pre 会做 dtype/shape/contiguous 检查，并分配 workspace：mixes + sqsum + pre.
然后把输入输出 tensor、workspace、shape、eps、sinkhorn_repeat 都放入 MhcPrePlannedMeta
*/
void *plan_pre(Tensor y,
               Tensor post,
               Tensor comb,
               const Tensor &residual,
               const Tensor &fn,
               const Tensor &hc_scale,
               const Tensor &hc_base,
               double rms_eps,
               double hc_pre_eps,
               double hc_sinkhorn_eps,
               int sinkhorn_repeat) {
    int64_t tokens = 0;
    int64_t hc = 0;
    int64_t hidden = 0;
    validate_pre_kernel_tensors(y, post, comb, residual, fn, hc_scale, hc_base, tokens, hc, hidden, "deepseek_v4_mhc_pre_kernel_");
    const int64_t mix_hc = (2 + hc) * hc;
    auto mixes = Tensor::empty({static_cast<size_t>(tokens), static_cast<size_t>(mix_hc)}, DataType::F32, residual->device());
    auto sqsum = Tensor::empty({static_cast<size_t>(tokens)}, DataType::F32, residual->device());
    auto pre = Tensor::empty({static_cast<size_t>(tokens), static_cast<size_t>(hc)}, DataType::F32, residual->device());
    return new MhcPrePlannedMeta{
        graph::GraphTensor(y),
        graph::GraphTensor(post),
        graph::GraphTensor(comb),
        graph::GraphTensor(residual),
        graph::GraphTensor(fn),
        graph::GraphTensor(hc_scale),
        graph::GraphTensor(hc_base),
        graph::GraphTensor(mixes),
        graph::GraphTensor(sqsum),
        graph::GraphTensor(pre),
        tokens,
        hc,
        hidden,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        sinkhorn_repeat};
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
    deepseek_v4_mhc_pre::launch_kernel(
        planned->y->data(),
        reinterpret_cast<float *>(planned->post->data()),
        reinterpret_cast<float *>(planned->comb->data()),
        planned->residual->data(),
        reinterpret_cast<const float *>(planned->fn->data()),
        reinterpret_cast<const float *>(planned->hc_scale->data()),
        reinterpret_cast<const float *>(planned->hc_base->data()),
        reinterpret_cast<float *>(planned->mixes->data()),
        reinterpret_cast<float *>(planned->sqsum->data()),
        reinterpret_cast<float *>(planned->pre->data()),
        planned->tokens,
        planned->hc,
        planned->hidden,
        planned->rms_eps,
        planned->hc_pre_eps,
        planned->hc_sinkhorn_eps,
        planned->sinkhorn_repeat,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_mhc_pre_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void cleanup_pre(void **planned_meta_ptr) {
    delete *reinterpret_cast<MhcPrePlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_mhc_pre_graph_impl

namespace deepseek_v4_mhc_pre_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4MhcPreKernel,
                                       &deepseek_v4_mhc_pre_graph_impl::plan_pre,
                                       &deepseek_v4_mhc_pre_graph_impl::run_pre,
                                       &deepseek_v4_mhc_pre_graph_impl::cleanup_pre);
} // namespace deepseek_v4_mhc_pre_register

/*
deepseek_v4_mhc_pre_ 当前内部调用链:

deepseek_v4_mhc_pre_
  -> deepseek_v4_mhc_pre_kernel_
    -> DeepseekV4MhcPreKernel::execute
      -> INFINICORE_GRAPH_OP_RECORD_OR_RUN
        -> 构造 DeepseekV4MhcPreKernel
          -> INFINICORE_GRAPH_OP_DISPATCH
            -> 调 plan_pre(...)
        -> 如果正在 graph recording：addGraphOperator
        -> 否则：op->run()
          -> run_pre(...)
            -> deepseek_v4_mhc_pre::launch_kernel(...)
              -> CUDA/HYGON kernel launch
*/
void deepseek_v4_mhc_pre_(Tensor y,
                          Tensor post,
                          Tensor comb,
                          const Tensor &residual,
                          const Tensor &fn,
                          const Tensor &hc_scale,
                          const Tensor &hc_base,
                          double rms_eps,
                          double hc_pre_eps,
                          double hc_sinkhorn_eps,
                          int sinkhorn_repeat) {
    deepseek_v4_mhc_pre_kernel_(y, post, comb, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, sinkhorn_repeat);
}

void deepseek_v4_mhc_pre_kernel_(Tensor y,
                                 Tensor post,
                                 Tensor comb,
                                 const Tensor &residual,
                                 const Tensor &fn,
                                 const Tensor &hc_scale,
                                 const Tensor &hc_base,
                                 double rms_eps,
                                 double hc_pre_eps,
                                 double hc_sinkhorn_eps,
                                 int sinkhorn_repeat) {
    DeepseekV4MhcPreKernel::execute(y, post, comb, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, sinkhorn_repeat);
}

} // namespace infinicore::op
