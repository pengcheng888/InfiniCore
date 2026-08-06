#include "infinicore/ops/deepseek_v4_mhc_pre.hpp"

#include "deepseek_v4_mhc_pre_kernel_v2.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4MhcPreKernelV2);

namespace {

bool mhc_graph_debug_enabled() {
    static const bool enabled = []() {
        const char *value = std::getenv("INFINICORE_GRAPH_DEBUG");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

bool mhc_tiled_splitk_enabled() {
    static const bool enabled = []() {
        const char *value = std::getenv("INFINICORE_MHC_PRE_V2_TILED_SPLITK");
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

DeepseekV4MhcPreKernelV2::DeepseekV4MhcPreKernelV2(Tensor y,
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

void DeepseekV4MhcPreKernelV2::execute(Tensor y,
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
                     "[infinicore graph] execute DeepseekV4MhcPreKernelV2 recording=%d\n",
                     context::isGraphRecording() ? 1 : 0);
    }
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4MhcPreKernelV2,
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

namespace deepseek_v4_mhc_pre_v2_graph_impl {

struct MhcPreV2PlannedMeta {
    graph::GraphTensor y;
    graph::GraphTensor post;
    graph::GraphTensor comb;
    graph::GraphTensor residual;
    graph::GraphTensor fn;
    graph::GraphTensor hc_scale;
    graph::GraphTensor hc_base;
    graph::GraphTensor partial_mixes;
    graph::GraphTensor partial_sqsum;
    int64_t tokens;
    int64_t hc;
    int64_t hidden;
    int split_k;
    int partial_stride;
    double rms_eps;
    double hc_pre_eps;
    double hc_sinkhorn_eps;
    int sinkhorn_repeat;
};

void validate_pre_v2_kernel_tensors(Tensor y,
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

void *plan_pre_v2(Tensor y,
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
    validate_pre_v2_kernel_tensors(y, post, comb, residual, fn, hc_scale, hc_base, tokens, hc, hidden, "deepseek_v4_mhc_pre_kernel_v2_");
    const int64_t k_size = hc * hidden;
    const int64_t mix_hc = (2 + hc) * hc;
    const bool use_tiled_splitk = mhc_tiled_splitk_enabled() && hc == 4 && hidden == 4096 && tokens >= 32 && tokens <= 2048;
    const int split_k = use_tiled_splitk ? 32 : ((hc == 4 && hidden == 4096) || tokens > 2048 || k_size < 4096 ? 1 : 32);
    const int partial_stride = use_tiled_splitk ? 32 : static_cast<int>(mix_hc);
    auto partial_mixes = Tensor::empty({static_cast<size_t>(split_k), static_cast<size_t>(tokens), static_cast<size_t>(partial_stride)}, DataType::F32, residual->device());
    auto partial_sqsum = Tensor::empty({static_cast<size_t>(split_k), static_cast<size_t>(tokens)}, DataType::F32, residual->device());
    return new MhcPreV2PlannedMeta{
        graph::GraphTensor(y),
        graph::GraphTensor(post),
        graph::GraphTensor(comb),
        graph::GraphTensor(residual),
        graph::GraphTensor(fn),
        graph::GraphTensor(hc_scale),
        graph::GraphTensor(hc_base),
        graph::GraphTensor(partial_mixes),
        graph::GraphTensor(partial_sqsum),
        tokens,
        hc,
        hidden,
        split_k,
        partial_stride,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        sinkhorn_repeat};
}

void run_pre_v2(void *planned_meta) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    auto *planned = reinterpret_cast<MhcPreV2PlannedMeta *>(planned_meta);
    if (mhc_graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[infinicore graph] run DeepseekV4MhcPreKernelV2 tokens=%ld hc=%ld hidden=%ld split_k=%d\n",
                     static_cast<long>(planned->tokens),
                     static_cast<long>(planned->hc),
                     static_cast<long>(planned->hidden),
                     planned->split_k);
    }
    deepseek_v4_mhc_pre_v2::launch_kernel(
        planned->y->data(),
        reinterpret_cast<float *>(planned->post->data()),
        reinterpret_cast<float *>(planned->comb->data()),
        planned->residual->data(),
        reinterpret_cast<const float *>(planned->fn->data()),
        reinterpret_cast<const float *>(planned->hc_scale->data()),
        reinterpret_cast<const float *>(planned->hc_base->data()),
        reinterpret_cast<float *>(planned->partial_mixes->data()),
        reinterpret_cast<float *>(planned->partial_sqsum->data()),
        planned->tokens,
        planned->hc,
        planned->hidden,
        planned->rms_eps,
        planned->hc_pre_eps,
        planned->hc_sinkhorn_eps,
        planned->sinkhorn_repeat,
        planned->split_k,
        planned->partial_stride,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_mhc_pre_kernel_v2_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void cleanup_pre_v2(void **planned_meta_ptr) {
    delete *reinterpret_cast<MhcPreV2PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_mhc_pre_v2_graph_impl

namespace deepseek_v4_mhc_pre_v2_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4MhcPreKernelV2,
                                       &deepseek_v4_mhc_pre_v2_graph_impl::plan_pre_v2,
                                       &deepseek_v4_mhc_pre_v2_graph_impl::run_pre_v2,
                                       &deepseek_v4_mhc_pre_v2_graph_impl::cleanup_pre_v2);
} // namespace deepseek_v4_mhc_pre_v2_register

void deepseek_v4_mhc_pre_kernel_v2_(Tensor y,
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
    DeepseekV4MhcPreKernelV2::execute(y, post, comb, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, sinkhorn_repeat);
}

} // namespace infinicore::op
