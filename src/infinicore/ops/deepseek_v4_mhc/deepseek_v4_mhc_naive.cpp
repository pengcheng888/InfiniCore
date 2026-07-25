#include "infinicore/ops/deepseek_v4_mhc_naive.hpp"

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
at::Tensor sinkhorn(at::Tensor comb, int sinkhorn_iters, double eps) {
    auto row_max = std::get<0>(comb.max(2, true));
    comb = at::exp(comb - row_max);
    comb = comb / comb.sum(2, true) + eps;
    comb = comb / (comb.sum(1, true) + eps);
    for (int i = 1; i < sinkhorn_iters; ++i) {
        comb = comb / (comb.sum(2, true) + eps);
        comb = comb / (comb.sum(1, true) + eps);
    }
    return comb;
}
#endif

} // namespace

void deepseek_v4_mhc_pre_naive_(Tensor y,
                                Tensor post,
                                Tensor comb,
                                const Tensor &x,
                                const Tensor &fn,
                                const Tensor &scale,
                                const Tensor &base,
                                double rms_eps,
                                double hc_eps,
                                int sinkhorn_iters) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(x, "deepseek_v4_mhc_pre_naive_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    if (x->ndim() != 3 || fn->ndim() != 2 || scale->ndim() != 1 || base->ndim() != 1) {
        throw std::runtime_error("deepseek_v4_mhc_pre_naive_ unexpected input rank.");
    }
    const int64_t tokens = static_cast<int64_t>(x->size(0));
    const int64_t hc = static_cast<int64_t>(x->size(1));
    const int64_t hidden = static_cast<int64_t>(x->size(2));
    const int64_t mix_hc = (2 + hc) * hc;
    if (fn->shape() != Shape{static_cast<size_t>(mix_hc), static_cast<size_t>(hc * hidden)} ||
        base->size(0) != static_cast<size_t>(mix_hc) || scale->size(0) != 3 ||
        y->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)} ||
        post->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc)} ||
        comb->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc), static_cast<size_t>(hc)}) {
        throw std::runtime_error("deepseek_v4_mhc_pre_naive_ shape mismatch.");
    }

    auto y_at = infinicore::adaptor::to_aten_tensor(y);
    auto post_at = infinicore::adaptor::to_aten_tensor(post);
    auto comb_at = infinicore::adaptor::to_aten_tensor(comb);
    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto fn_at = infinicore::adaptor::to_aten_tensor(fn).to(at::kFloat);
    auto scale_at = infinicore::adaptor::to_aten_tensor(scale).to(at::kFloat);
    auto base_at = infinicore::adaptor::to_aten_tensor(base).to(at::kFloat);

    auto x_flat = x_at.reshape({tokens, hc * hidden}).to(at::kFloat);
    auto rsqrt = at::rsqrt(x_flat.square().mean(-1, true) + rms_eps);
    auto mixes = at::matmul(x_flat, fn_at.transpose(0, 1)) * rsqrt;
    auto pre = at::sigmoid(mixes.slice(1, 0, hc) * scale_at[0] + base_at.slice(0, 0, hc)) + hc_eps;
    auto post_result = 2.0 * at::sigmoid(mixes.slice(1, hc, 2 * hc) * scale_at[1] + base_at.slice(0, hc, 2 * hc));
    auto comb_result = mixes.slice(1, 2 * hc, mix_hc).reshape({tokens, hc, hc}) * scale_at[2] +
                       base_at.slice(0, 2 * hc, mix_hc).reshape({1, hc, hc});
    comb_result = sinkhorn(comb_result, sinkhorn_iters, hc_eps);
    auto y_result = (pre.unsqueeze(-1) * x_at.to(at::kFloat)).sum(1).to(y_at.scalar_type());

    y_at.copy_(y_result);
    post_at.copy_(post_result.to(post_at.scalar_type()));
    comb_at.copy_(comb_result.to(comb_at.scalar_type()));
#else
    (void)y;
    (void)post;
    (void)comb;
    (void)x;
    (void)fn;
    (void)scale;
    (void)base;
    (void)rms_eps;
    (void)hc_eps;
    (void)sinkhorn_iters;
    throw std::runtime_error("deepseek_v4_mhc_pre_naive_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_mhc_post_naive_(Tensor y,
                                 const Tensor &x,
                                 const Tensor &residual,
                                 const Tensor &post,
                                 const Tensor &comb) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(x, "deepseek_v4_mhc_post_naive_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto y_at = infinicore::adaptor::to_aten_tensor(y);
    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto residual_at = infinicore::adaptor::to_aten_tensor(residual);
    auto post_at = infinicore::adaptor::to_aten_tensor(post);
    auto comb_at = infinicore::adaptor::to_aten_tensor(comb);
    auto result = post_at.unsqueeze(-1) * x_at.to(at::kFloat).unsqueeze(1) +
                  at::matmul(comb_at.transpose(1, 2), residual_at.to(at::kFloat));
    y_at.copy_(result.to(y_at.scalar_type()));
#else
    (void)y;
    (void)x;
    (void)residual;
    (void)post;
    (void)comb;
    throw std::runtime_error("deepseek_v4_mhc_post_naive_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_mhc_head_naive_(Tensor y,
                                 const Tensor &x,
                                 const Tensor &fn,
                                 const Tensor &scale,
                                 const Tensor &base,
                                 double rms_eps,
                                 double hc_eps) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(x, "deepseek_v4_mhc_head_naive_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    if (x->ndim() != 3 || fn->ndim() != 2 || scale->ndim() != 1 || base->ndim() != 1) {
        throw std::runtime_error("deepseek_v4_mhc_head_naive_ unexpected input rank.");
    }
    const int64_t tokens = static_cast<int64_t>(x->size(0));
    const int64_t hc = static_cast<int64_t>(x->size(1));
    const int64_t hidden = static_cast<int64_t>(x->size(2));
    if (fn->shape() != Shape{static_cast<size_t>(hc), static_cast<size_t>(hc * hidden)} ||
        base->size(0) != static_cast<size_t>(hc) || scale->size(0) != 1 ||
        y->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)}) {
        throw std::runtime_error("deepseek_v4_mhc_head_naive_ shape mismatch.");
    }

    auto y_at = infinicore::adaptor::to_aten_tensor(y);
    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto fn_at = infinicore::adaptor::to_aten_tensor(fn).to(at::kFloat);
    auto scale_at = infinicore::adaptor::to_aten_tensor(scale).to(at::kFloat);
    auto base_at = infinicore::adaptor::to_aten_tensor(base).to(at::kFloat);
    auto x_flat = x_at.reshape({tokens, hc * hidden}).to(at::kFloat);
    auto rsqrt = at::rsqrt(x_flat.square().mean(-1, true) + rms_eps);
    auto mixes = at::matmul(x_flat, fn_at.transpose(0, 1)) * rsqrt;
    auto pre = at::sigmoid(mixes * scale_at[0] + base_at) + hc_eps;
    auto result = (pre.unsqueeze(-1) * x_at.to(at::kFloat)).sum(1);
    y_at.copy_(result.to(y_at.scalar_type()));
#else
    (void)y;
    (void)x;
    (void)fn;
    (void)scale;
    (void)base;
    (void)rms_eps;
    (void)hc_eps;
    throw std::runtime_error("deepseek_v4_mhc_head_naive_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
