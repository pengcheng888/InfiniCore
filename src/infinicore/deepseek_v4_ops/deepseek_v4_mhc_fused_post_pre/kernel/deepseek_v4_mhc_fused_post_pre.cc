#include "infinicore/ops/deepseek_v4_mhc_fused_post_pre.hpp"

#include "deepseek_v4_mhc_fused_post_pre_kernel.hpp"

#include "../../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4MhcFusedPostPre);

} // namespace deepseek_v4

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

bool use_hc4_hidden4096_fused_path(const Tensor &residual) {
    return residual->ndim() == 3
        && residual->size(0) > 0
        && residual->size(1) == 4
        && residual->size(2) == 4096;
}

} // namespace

namespace deepseek_v4 {

DeepseekV4MhcFusedPostPre::DeepseekV4MhcFusedPostPre(Tensor residual_cur,
                                                     Tensor post_mix_cur,
                                                     Tensor comb_mix_cur,
                                                     Tensor layer_input_cur,
                                                     const Tensor &x,
                                                     const Tensor &residual,
                                                     const Tensor &post_layer_mix,
                                                     const Tensor &comb_res_mix,
                                                     const Tensor &fn,
                                                     const Tensor &hc_scale,
                                                     const Tensor &hc_base,
                                                     double rms_eps,
                                                     double hc_pre_eps,
                                                     double hc_sinkhorn_eps,
                                                     double hc_post_mult_value,
                                                     int sinkhorn_repeat,
                                                     const Tensor &norm_weight,
                                                     double norm_eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, norm_weight);
    INFINICORE_GRAPH_OP_DISPATCH(residual_cur->device().getType(),
                                 residual_cur,
                                 post_mix_cur,
                                 comb_mix_cur,
                                 layer_input_cur,
                                 x,
                                 residual,
                                 post_layer_mix,
                                 comb_res_mix,
                                 fn,
                                 hc_scale,
                                 hc_base,
                                 rms_eps,
                                 hc_pre_eps,
                                 hc_sinkhorn_eps,
                                 hc_post_mult_value,
                                 sinkhorn_repeat,
                                 norm_weight,
                                 norm_eps);
}

void DeepseekV4MhcFusedPostPre::execute(Tensor residual_cur,
                                        Tensor post_mix_cur,
                                        Tensor comb_mix_cur,
                                        Tensor layer_input_cur,
                                        const Tensor &x,
                                        const Tensor &residual,
                                        const Tensor &post_layer_mix,
                                        const Tensor &comb_res_mix,
                                        const Tensor &fn,
                                        const Tensor &hc_scale,
                                        const Tensor &hc_base,
                                        double rms_eps,
                                        double hc_pre_eps,
                                        double hc_sinkhorn_eps,
                                        double hc_post_mult_value,
                                        int sinkhorn_repeat,
                                        const Tensor &norm_weight,
                                        double norm_eps) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4MhcFusedPostPre,
                                      residual_cur,
                                      post_mix_cur,
                                      comb_mix_cur,
                                      layer_input_cur,
                                      x,
                                      residual,
                                      post_layer_mix,
                                      comb_res_mix,
                                      fn,
                                      hc_scale,
                                      hc_base,
                                      rms_eps,
                                      hc_pre_eps,
                                      hc_sinkhorn_eps,
                                      hc_post_mult_value,
                                      sinkhorn_repeat,
                                      norm_weight,
                                      norm_eps);
}

namespace deepseek_v4_mhc_fused_post_pre_impl {

struct MhcFusedPostPrePlannedMeta {
    graph::GraphTensor residual_cur;
    graph::GraphTensor post_mix_cur;
    graph::GraphTensor comb_mix_cur;
    graph::GraphTensor layer_input_cur;
    graph::GraphTensor x;
    graph::GraphTensor residual;
    graph::GraphTensor post_layer_mix;
    graph::GraphTensor comb_res_mix;
    graph::GraphTensor fn;
    graph::GraphTensor hc_scale;
    graph::GraphTensor hc_base;
    graph::GraphTensor norm_weight;
    graph::GraphTensor mixes;
    graph::GraphTensor sqsum;
    graph::GraphTensor pre;
    graph::GraphTensor mixes_partial;
    graph::GraphTensor sqsum_partial;
    int64_t tokens;
    int64_t hc;
    int64_t hidden;
    double rms_eps;
    double norm_eps;
    double hc_pre_eps;
    double hc_sinkhorn_eps;
    double hc_post_mult_value;
    int sinkhorn_repeat;
};

void validate_tensors(Tensor residual_cur,
                      Tensor post_mix_cur,
                      Tensor comb_mix_cur,
                      Tensor layer_input_cur,
                      const Tensor &x,
                      const Tensor &residual,
                      const Tensor &post_layer_mix,
                      const Tensor &comb_res_mix,
                      const Tensor &fn,
                      const Tensor &hc_scale,
                      const Tensor &hc_base,
                      const Tensor &norm_weight,
                      double hc_post_mult_value,
                      int64_t &tokens,
                      int64_t &hc,
                      int64_t &hidden,
                      const char *op_name) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(x, op_name);
    check_dtype(x, DataType::BF16, op_name, "x");
    check_dtype(residual, DataType::BF16, op_name, "residual");
    check_dtype(residual_cur, DataType::BF16, op_name, "residual_cur");
    check_dtype(layer_input_cur, DataType::BF16, op_name, "layer_input_cur");
    check_dtype(post_layer_mix, DataType::F32, op_name, "post_layer_mix");
    check_dtype(comb_res_mix, DataType::F32, op_name, "comb_res_mix");
    check_dtype(post_mix_cur, DataType::F32, op_name, "post_mix_cur");
    check_dtype(comb_mix_cur, DataType::F32, op_name, "comb_mix_cur");
    check_dtype(fn, DataType::F32, op_name, "fn");
    check_dtype(hc_scale, DataType::F32, op_name, "hc_scale");
    check_dtype(hc_base, DataType::F32, op_name, "hc_base");
    check_dtype(norm_weight, DataType::BF16, op_name, "norm_weight");

    if (hc_post_mult_value != 2.0) {
        throw std::runtime_error(std::string(op_name) + " currently expects hc_post_mult_value == 2.0.");
    }
    if (x->ndim() != 2 || residual->ndim() != 3 || post_layer_mix->ndim() != 2 || comb_res_mix->ndim() != 3 || fn->ndim() != 2 || hc_scale->ndim() != 1 || hc_base->ndim() != 1 || norm_weight->ndim() != 1) {
        throw std::runtime_error(std::string(op_name) + " unexpected input rank.");
    }
    tokens = static_cast<int64_t>(residual->size(0));
    hc = static_cast<int64_t>(residual->size(1));
    hidden = static_cast<int64_t>(residual->size(2));
    const int64_t mix_hc = (2 + hc) * hc;
    if (hc > 16) {
        throw std::runtime_error(std::string(op_name) + " supports hc <= 16.");
    }
    if (x->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)}
        || residual_cur->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc), static_cast<size_t>(hidden)}
        || layer_input_cur->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)}
        || post_layer_mix->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc)}
        || comb_res_mix->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc), static_cast<size_t>(hc)}
        || post_mix_cur->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc)}
        || comb_mix_cur->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc), static_cast<size_t>(hc)}
        || fn->shape() != Shape{static_cast<size_t>(mix_hc), static_cast<size_t>(hc * hidden)}
        || hc_base->size(0) != static_cast<size_t>(mix_hc)
        || hc_scale->size(0) != 3
        || norm_weight->size(0) != static_cast<size_t>(hidden)) {
        throw std::runtime_error(std::string(op_name) + " shape mismatch.");
    }
    check_contiguous_tensor(residual_cur, op_name, "residual_cur");
    check_contiguous_tensor(post_mix_cur, op_name, "post_mix_cur");
    check_contiguous_tensor(comb_mix_cur, op_name, "comb_mix_cur");
    check_contiguous_tensor(layer_input_cur, op_name, "layer_input_cur");
    check_contiguous_tensor(x, op_name, "x");
    check_contiguous_tensor(residual, op_name, "residual");
    check_contiguous_tensor(post_layer_mix, op_name, "post_layer_mix");
    check_contiguous_tensor(comb_res_mix, op_name, "comb_res_mix");
    check_contiguous_tensor(fn, op_name, "fn");
    check_contiguous_tensor(hc_scale, op_name, "hc_scale");
    check_contiguous_tensor(hc_base, op_name, "hc_base");
    check_contiguous_tensor(norm_weight, op_name, "norm_weight");
#else
    (void)residual_cur;
    (void)post_mix_cur;
    (void)comb_mix_cur;
    (void)layer_input_cur;
    (void)x;
    (void)residual;
    (void)post_layer_mix;
    (void)comb_res_mix;
    (void)fn;
    (void)hc_scale;
    (void)hc_base;
    (void)norm_weight;
    (void)hc_post_mult_value;
    (void)tokens;
    (void)hc;
    (void)hidden;
    throw std::runtime_error(std::string(op_name) + " requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void *plan(Tensor residual_cur,
           Tensor post_mix_cur,
           Tensor comb_mix_cur,
           Tensor layer_input_cur,
           const Tensor &x,
           const Tensor &residual,
           const Tensor &post_layer_mix,
           const Tensor &comb_res_mix,
           const Tensor &fn,
           const Tensor &hc_scale,
           const Tensor &hc_base,
           double rms_eps,
           double hc_pre_eps,
           double hc_sinkhorn_eps,
           double hc_post_mult_value,
           int sinkhorn_repeat,
           const Tensor &norm_weight,
           double norm_eps) {
    int64_t tokens = 0;
    int64_t hc = 0;
    int64_t hidden = 0;
    validate_tensors(residual_cur,
                     post_mix_cur,
                     comb_mix_cur,
                     layer_input_cur,
                     x,
                     residual,
                     post_layer_mix,
                     comb_res_mix,
                     fn,
                     hc_scale,
                     hc_base,
                     norm_weight,
                     hc_post_mult_value,
                     tokens,
                     hc,
                     hidden,
                     "deepseek_v4_mhc_fused_post_pre_kernel_");
    const int64_t mix_hc = (2 + hc) * hc;
    auto mixes = Tensor::empty({static_cast<size_t>(tokens), static_cast<size_t>(mix_hc)}, DataType::F32, x->device());
    auto sqsum = Tensor::empty({static_cast<size_t>(tokens)}, DataType::F32, x->device());
    auto pre = Tensor::empty({static_cast<size_t>(tokens), static_cast<size_t>(hc)}, DataType::F32, x->device());
    int64_t small_path_split_k = 1;
    if (hc == 4 && hidden == 4096 && tokens < 128) {
        small_path_split_k = 8;
    }
    auto mixes_partial = Tensor::empty({static_cast<size_t>(small_path_split_k), static_cast<size_t>(tokens), static_cast<size_t>(mix_hc)}, DataType::F32, x->device());
    auto sqsum_partial = Tensor::empty({static_cast<size_t>(small_path_split_k), static_cast<size_t>(tokens)}, DataType::F32, x->device());
    return new MhcFusedPostPrePlannedMeta{
        graph::GraphTensor(residual_cur),
        graph::GraphTensor(post_mix_cur),
        graph::GraphTensor(comb_mix_cur),
        graph::GraphTensor(layer_input_cur),
        graph::GraphTensor(x),
        graph::GraphTensor(residual),
        graph::GraphTensor(post_layer_mix),
        graph::GraphTensor(comb_res_mix),
        graph::GraphTensor(fn),
        graph::GraphTensor(hc_scale),
        graph::GraphTensor(hc_base),
        graph::GraphTensor(norm_weight),
        graph::GraphTensor(mixes),
        graph::GraphTensor(sqsum),
        graph::GraphTensor(pre),
        graph::GraphTensor(mixes_partial),
        graph::GraphTensor(sqsum_partial),
        tokens,
        hc,
        hidden,
        rms_eps,
        norm_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat};
}

void run(void *planned_meta) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    auto *planned = reinterpret_cast<MhcFusedPostPrePlannedMeta *>(planned_meta);
    deepseek_v4_mhc_fused_post_pre::launch_kernel(
        planned->residual_cur->data(),
        reinterpret_cast<float *>(planned->post_mix_cur->data()),
        reinterpret_cast<float *>(planned->comb_mix_cur->data()),
        planned->layer_input_cur->data(),
        planned->x->data(),
        planned->residual->data(),
        reinterpret_cast<const float *>(planned->post_layer_mix->data()),
        reinterpret_cast<const float *>(planned->comb_res_mix->data()),
        reinterpret_cast<const float *>(planned->fn->data()),
        reinterpret_cast<const float *>(planned->hc_scale->data()),
        reinterpret_cast<const float *>(planned->hc_base->data()),
        reinterpret_cast<float *>(planned->mixes->data()),
        reinterpret_cast<float *>(planned->sqsum->data()),
        reinterpret_cast<float *>(planned->pre->data()),
        reinterpret_cast<float *>(planned->mixes_partial->data()),
        reinterpret_cast<float *>(planned->sqsum_partial->data()),
        planned->tokens,
        planned->hc,
        planned->hidden,
        planned->rms_eps,
        planned->hc_pre_eps,
        planned->hc_sinkhorn_eps,
        planned->hc_post_mult_value,
        planned->sinkhorn_repeat,
        planned->norm_weight->data(),
        planned->norm_eps,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_mhc_fused_post_pre_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<MhcFusedPostPrePlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_mhc_fused_post_pre_impl

namespace deepseek_v4_mhc_fused_post_pre_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4MhcFusedPostPre,
                                       &deepseek_v4_mhc_fused_post_pre_impl::plan,
                                       &deepseek_v4_mhc_fused_post_pre_impl::run,
                                       &deepseek_v4_mhc_fused_post_pre_impl::cleanup);
} // namespace deepseek_v4_mhc_fused_post_pre_register

} // namespace deepseek_v4

// **不同点**
// 主要不同在“怎么拆 kernel、怎么分 token 路径”。

// SGLang 当前逻辑：

// ```text
// num_tokens <= 32:
//     mhc_fused_post_pre_fma_tilelang
//     mhc_pre_big_fuse_with_norm_tilelang

// num_tokens > 32:
//     mhc_post_tilelang
//     deep_gemm.tf32_hc_prenorm_gemm
//       或 mhc_pre_gemm_sqrsum_tilelang
//     mhc_pre_big_fuse_with_norm_tilelang
// ```

// InfiniCore 当前逻辑：

// ```text
// tokens < 128:
//     mhc_fused_post_pre_fma_hc4_hidden4096_mix3_split8_kernel
//     mhc_pre_big_fuse_with_norm_hc4_hidden4096_split8_kernel

// tokens >= 128:
//     mhc_post_mix_sqsum_hc4_hidden4096_all24_kernel
//     mhc_pre_finalize_y_norm_hc4_hidden4096_kernel
// ```
void deepseek_v4_mhc_fused_post_pre_kernel_(Tensor residual_cur,
                                            Tensor post_mix_cur,
                                            Tensor comb_mix_cur,
                                            Tensor layer_input_cur,
                                            const Tensor &x,
                                            const Tensor &residual,
                                            const Tensor &post_layer_mix,
                                            const Tensor &comb_res_mix,
                                            const Tensor &fn,
                                            const Tensor &hc_scale,
                                            const Tensor &hc_base,
                                            double rms_eps,
                                            double hc_pre_eps,
                                            double hc_sinkhorn_eps,
                                            double hc_post_mult_value,
                                            int sinkhorn_repeat,
                                            const Tensor &norm_weight,
                                            double norm_eps) {
    if (!use_hc4_hidden4096_fused_path(residual)) {
        throw std::runtime_error("deepseek_v4_mhc_fused_post_pre_kernel_ expects standard fused shape [tokens, 4, 4096].");
    }

    deepseek_v4::DeepseekV4MhcFusedPostPre::execute(residual_cur,
                                                    post_mix_cur,
                                                    comb_mix_cur,
                                                    layer_input_cur,
                                                    x,
                                                    residual,
                                                    post_layer_mix,
                                                    comb_res_mix,
                                                    fn,
                                                    hc_scale,
                                                    hc_base,
                                                    rms_eps,
                                                    hc_pre_eps,
                                                    hc_sinkhorn_eps,
                                                    hc_post_mult_value,
                                                    sinkhorn_repeat,
                                                    norm_weight,
                                                    norm_eps);
}

} // namespace infinicore::op
