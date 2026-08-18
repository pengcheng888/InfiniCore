#include "infinicore/ops/deepseek_v4_mhc_pre.hpp"

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

namespace infinicore::op {
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
at::Tensor unweighted_rmsnorm(const at::Tensor &x, double eps) {
    return x * at::rsqrt(x.square().mean(-1, true) + eps);
}

at::Tensor sinkhorn(at::Tensor comb, int sinkhorn_repeat, double eps) {
    comb = at::softmax(comb, 2) + eps;
    comb = comb / (comb.sum(1, true) + eps);
    for (int i = 1; i < sinkhorn_repeat; ++i) {
        comb = comb / (comb.sum(2, true) + eps);
        comb = comb / (comb.sum(1, true) + eps);
    }
    return comb;
}
#endif

} // namespace

void deepseek_v4_mhc_pre_aten_(Tensor y,
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
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(residual, "deepseek_v4_mhc_pre_aten_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    if (residual->ndim() != 3 || fn->ndim() != 2 || hc_scale->ndim() != 1 || hc_base->ndim() != 1) {
        throw std::runtime_error("deepseek_v4_mhc_pre_aten_ unexpected input rank.");
    }
    const int64_t tokens = static_cast<int64_t>(residual->size(0));
    const int64_t hc = static_cast<int64_t>(residual->size(1));
    const int64_t hidden = static_cast<int64_t>(residual->size(2));
    const int64_t mix_hc = (2 + hc) * hc;
    if (fn->shape() != Shape{static_cast<size_t>(mix_hc), static_cast<size_t>(hc * hidden)}
        || hc_base->size(0) != static_cast<size_t>(mix_hc)
        || hc_scale->size(0) != 3
        || y->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)}
        || post->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc)}
        || comb->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc), static_cast<size_t>(hc)}) {
        throw std::runtime_error("deepseek_v4_mhc_pre_aten_ shape mismatch.");
    }

    auto y_at = infinicore::adaptor::to_aten_tensor(y);
    auto post_at = infinicore::adaptor::to_aten_tensor(post);
    auto comb_at = infinicore::adaptor::to_aten_tensor(comb);
    auto residual_at = infinicore::adaptor::to_aten_tensor(residual);
    auto fn_at = infinicore::adaptor::to_aten_tensor(fn).to(at::kFloat);
    auto hc_scale_at = infinicore::adaptor::to_aten_tensor(hc_scale).to(at::kFloat);
    auto hc_base_at = infinicore::adaptor::to_aten_tensor(hc_base).to(at::kFloat);

    auto x_flat = residual_at.reshape({tokens, hc * hidden}).to(at::kFloat);
    auto flat = unweighted_rmsnorm(x_flat, rms_eps);

    auto mixes = at::matmul(flat, fn_at.transpose(0, 1));
    auto mix_parts = mixes.split_with_sizes({hc, hc, hc * hc}, 1);
    auto pre_w = mix_parts[0];
    auto post_w = mix_parts[1];
    auto comb_w = mix_parts[2];

    auto base_parts = hc_base_at.split_with_sizes({hc, hc, hc * hc}, 0);
    auto pre_b = base_parts[0];
    auto post_b = base_parts[1];
    auto comb_b = base_parts[2];

    auto scale_parts = hc_scale_at.unbind(0);
    auto pre_scale = scale_parts[0];
    auto post_scale = scale_parts[1];
    auto comb_scale = scale_parts[2];

    auto pre = at::sigmoid(pre_w * pre_scale + pre_b) + hc_pre_eps;
    auto post_result = 2.0 * at::sigmoid(post_w * post_scale + post_b);
    auto comb_result = comb_w.reshape({tokens, hc, hc}) * comb_scale + comb_b.reshape({1, hc, hc});
    comb_result = sinkhorn(comb_result, sinkhorn_repeat, hc_sinkhorn_eps);
    auto y_result = (pre.unsqueeze(-1) * residual_at.to(at::kFloat)).sum(1).to(y_at.scalar_type());

    y_at.copy_(y_result);
    post_at.copy_(post_result.to(post_at.scalar_type()));
    comb_at.copy_(comb_result.to(comb_at.scalar_type()));
#else
    (void)y;
    (void)post;
    (void)comb;
    (void)residual;
    (void)fn;
    (void)hc_scale;
    (void)hc_base;
    (void)rms_eps;
    (void)hc_pre_eps;
    (void)hc_sinkhorn_eps;
    (void)sinkhorn_repeat;
    throw std::runtime_error("deepseek_v4_mhc_pre_aten_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
