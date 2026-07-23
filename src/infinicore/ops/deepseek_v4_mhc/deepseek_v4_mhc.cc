#include "infinicore/ops/deepseek_v4_mhc.hpp"

#include "deepseek_v4_mhc_kernel.hpp"

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

void check_contiguous_aten(const at::Tensor &tensor, const char *op_name, const char *arg_name) {
    if (!tensor.is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensor: " + arg_name);
    }
}

void check_dtype(const Tensor &tensor, DataType dtype, const char *op_name, const char *arg_name) {
    if (tensor->dtype() != dtype) {
        throw std::runtime_error(std::string(op_name) + " unexpected dtype for " + arg_name + ": expected " + toString(dtype) + ", got " + toString(tensor->dtype()));
    }
}

void *current_accelerator_stream() {
#if defined(ENABLE_HYGON_API)
    return reinterpret_cast<void *>(infinicore::adaptor::get_hip_stream().stream());
#else
    return reinterpret_cast<void *>(infinicore::adaptor::get_cuda_stream().stream());
#endif
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
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    const char *op_name = "deepseek_v4_mhc_pre_kernel_";
    check_accelerator_tensor(x, op_name);
    check_dtype(x, DataType::BF16, op_name, "x");
    check_dtype(y, DataType::BF16, op_name, "y");
    check_dtype(post, DataType::F32, op_name, "post");
    check_dtype(comb, DataType::F32, op_name, "comb");
    check_dtype(fn, DataType::F32, op_name, "fn");
    check_dtype(scale, DataType::F32, op_name, "scale");
    check_dtype(base, DataType::F32, op_name, "base");

#if defined(ENABLE_HYGON_API)
    auto stream_guard = infinicore::adaptor::get_hip_stream();
    c10::hip::HIPStreamGuard guard(stream_guard);
#else
    auto stream_guard = infinicore::adaptor::get_cuda_stream();
    c10::cuda::CUDAStreamGuard guard(stream_guard);
#endif

    if (x->ndim() != 3 || fn->ndim() != 2 || scale->ndim() != 1 || base->ndim() != 1) {
        throw std::runtime_error("deepseek_v4_mhc_pre_kernel_ unexpected input rank.");
    }
    const int64_t tokens = static_cast<int64_t>(x->size(0));
    const int64_t hc = static_cast<int64_t>(x->size(1));
    const int64_t hidden = static_cast<int64_t>(x->size(2));
    const int64_t mix_hc = (2 + hc) * hc;
    if (hc > 16) {
        throw std::runtime_error("deepseek_v4_mhc_pre_kernel_ supports hc <= 16.");
    }
    if (fn->shape() != Shape{static_cast<size_t>(mix_hc), static_cast<size_t>(hc * hidden)} ||
        base->size(0) != static_cast<size_t>(mix_hc) || scale->size(0) != 3 ||
        y->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)} ||
        post->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc)} ||
        comb->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc), static_cast<size_t>(hc)}) {
        throw std::runtime_error("deepseek_v4_mhc_pre_kernel_ shape mismatch.");
    }

    auto y_at = infinicore::adaptor::to_aten_tensor(y);
    auto post_at = infinicore::adaptor::to_aten_tensor(post);
    auto comb_at = infinicore::adaptor::to_aten_tensor(comb);
    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto fn_at = infinicore::adaptor::to_aten_tensor(fn);
    auto scale_at = infinicore::adaptor::to_aten_tensor(scale);
    auto base_at = infinicore::adaptor::to_aten_tensor(base);
    check_contiguous_aten(y_at, op_name, "y");
    check_contiguous_aten(post_at, op_name, "post");
    check_contiguous_aten(comb_at, op_name, "comb");
    check_contiguous_aten(x_at, op_name, "x");
    check_contiguous_aten(fn_at, op_name, "fn");
    check_contiguous_aten(scale_at, op_name, "scale");
    check_contiguous_aten(base_at, op_name, "base");

    auto options = fn_at.options().dtype(at::kFloat);
    auto mixes = at::empty({tokens, mix_hc}, options);
    auto sqsum = at::empty({tokens}, options);
    auto pre = at::empty({tokens, hc}, options);
    deepseek_v4_mhc::launch_pre_kernel(
        y_at.data_ptr(),
        post_at.data_ptr<float>(),
        comb_at.data_ptr<float>(),
        x_at.data_ptr(),
        fn_at.data_ptr<float>(),
        scale_at.data_ptr<float>(),
        base_at.data_ptr<float>(),
        mixes.data_ptr<float>(),
        sqsum.data_ptr<float>(),
        pre.data_ptr<float>(),
        tokens,
        hc,
        hidden,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        current_accelerator_stream());
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
    throw std::runtime_error("deepseek_v4_mhc_pre_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
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

void deepseek_v4_mhc_post_kernel_(Tensor y,
                           const Tensor &x,
                           const Tensor &residual,
                           const Tensor &post,
                           const Tensor &comb) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    const char *op_name = "deepseek_v4_mhc_post_kernel_";
    check_accelerator_tensor(x, op_name);
    check_dtype(x, DataType::BF16, op_name, "x");
    check_dtype(residual, DataType::BF16, op_name, "residual");
    check_dtype(y, DataType::BF16, op_name, "y");
    check_dtype(post, DataType::F32, op_name, "post");
    check_dtype(comb, DataType::F32, op_name, "comb");

#if defined(ENABLE_HYGON_API)
    auto stream_guard = infinicore::adaptor::get_hip_stream();
    c10::hip::HIPStreamGuard guard(stream_guard);
#else
    auto stream_guard = infinicore::adaptor::get_cuda_stream();
    c10::cuda::CUDAStreamGuard guard(stream_guard);
#endif

    if (x->ndim() != 2 || residual->ndim() != 3 || post->ndim() != 2 || comb->ndim() != 3) {
        throw std::runtime_error("deepseek_v4_mhc_post_kernel_ unexpected input rank.");
    }
    const int64_t tokens = static_cast<int64_t>(residual->size(0));
    const int64_t hc = static_cast<int64_t>(residual->size(1));
    const int64_t hidden = static_cast<int64_t>(residual->size(2));
    if (x->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)} ||
        y->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc), static_cast<size_t>(hidden)} ||
        post->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc)} ||
        comb->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hc), static_cast<size_t>(hc)}) {
        throw std::runtime_error("deepseek_v4_mhc_post_kernel_ shape mismatch.");
    }

    auto y_at = infinicore::adaptor::to_aten_tensor(y);
    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto residual_at = infinicore::adaptor::to_aten_tensor(residual);
    auto post_at = infinicore::adaptor::to_aten_tensor(post);
    auto comb_at = infinicore::adaptor::to_aten_tensor(comb);
    check_contiguous_aten(y_at, op_name, "y");
    check_contiguous_aten(x_at, op_name, "x");
    check_contiguous_aten(residual_at, op_name, "residual");
    check_contiguous_aten(post_at, op_name, "post");
    check_contiguous_aten(comb_at, op_name, "comb");

    deepseek_v4_mhc::launch_post_kernel(
        y_at.data_ptr(),
        x_at.data_ptr(),
        residual_at.data_ptr(),
        post_at.data_ptr<float>(),
        comb_at.data_ptr<float>(),
        tokens,
        hc,
        hidden,
        current_accelerator_stream());
#else
    (void)y;
    (void)x;
    (void)residual;
    (void)post;
    (void)comb;
    throw std::runtime_error("deepseek_v4_mhc_post_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
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


void deepseek_v4_mhc_head_kernel_(Tensor y,
                           const Tensor &x,
                           const Tensor &fn,
                           const Tensor &scale,
                           const Tensor &base,
                           double rms_eps,
                           double hc_eps) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    const char *op_name = "deepseek_v4_mhc_head_kernel_";
    check_accelerator_tensor(x, op_name);
    check_dtype(x, DataType::BF16, op_name, "x");
    check_dtype(y, DataType::BF16, op_name, "y");
    check_dtype(fn, DataType::F32, op_name, "fn");
    check_dtype(scale, DataType::F32, op_name, "scale");
    check_dtype(base, DataType::F32, op_name, "base");

#if defined(ENABLE_HYGON_API)
    auto stream_guard = infinicore::adaptor::get_hip_stream();
    c10::hip::HIPStreamGuard guard(stream_guard);
#else
    auto stream_guard = infinicore::adaptor::get_cuda_stream();
    c10::cuda::CUDAStreamGuard guard(stream_guard);
#endif

    if (x->ndim() != 3 || fn->ndim() != 2 || scale->ndim() != 1 || base->ndim() != 1) {
        throw std::runtime_error("deepseek_v4_mhc_head_kernel_ unexpected input rank.");
    }
    const int64_t tokens = static_cast<int64_t>(x->size(0));
    const int64_t hc = static_cast<int64_t>(x->size(1));
    const int64_t hidden = static_cast<int64_t>(x->size(2));
    if (fn->shape() != Shape{static_cast<size_t>(hc), static_cast<size_t>(hc * hidden)} ||
        base->size(0) != static_cast<size_t>(hc) || scale->size(0) != 1 ||
        y->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(hidden)}) {
        throw std::runtime_error("deepseek_v4_mhc_head_kernel_ shape mismatch.");
    }

    auto y_at = infinicore::adaptor::to_aten_tensor(y);
    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    auto fn_at = infinicore::adaptor::to_aten_tensor(fn);
    auto scale_at = infinicore::adaptor::to_aten_tensor(scale);
    auto base_at = infinicore::adaptor::to_aten_tensor(base);
    check_contiguous_aten(y_at, op_name, "y");
    check_contiguous_aten(x_at, op_name, "x");
    check_contiguous_aten(fn_at, op_name, "fn");
    check_contiguous_aten(scale_at, op_name, "scale");
    check_contiguous_aten(base_at, op_name, "base");

    auto options = fn_at.options().dtype(at::kFloat);
    auto mixes = at::empty({tokens, hc}, options);
    auto sqsum = at::empty({tokens}, options);
    deepseek_v4_mhc::launch_head_kernel(
        y_at.data_ptr(),
        x_at.data_ptr(),
        fn_at.data_ptr<float>(),
        scale_at.data_ptr<float>(),
        base_at.data_ptr<float>(),
        mixes.data_ptr<float>(),
        sqsum.data_ptr<float>(),
        tokens,
        hc,
        hidden,
        rms_eps,
        hc_eps,
        current_accelerator_stream());
#else
    (void)y;
    (void)x;
    (void)fn;
    (void)scale;
    (void)base;
    (void)rms_eps;
    (void)hc_eps;
    throw std::runtime_error("deepseek_v4_mhc_head_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
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
