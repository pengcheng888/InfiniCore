#include "infinicore/ops/deepseek_v4_fused_norm_rope_inplace.hpp"

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

namespace infinicore::op {

void deepseek_v4_fused_norm_rope_inplace_naive_(Tensor input,
                                                const Tensor &norm_weight,
                                                float epsilon,
                                                const Tensor &freqs_cis,
                                                const Tensor &positions) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    auto x = infinicore::adaptor::to_aten_tensor(input);
    auto weight = infinicore::adaptor::to_aten_tensor(norm_weight);
    auto freqs = infinicore::adaptor::to_aten_tensor(freqs_cis);
    auto pos = infinicore::adaptor::to_aten_tensor(positions).to(at::kLong);
    auto x_fp32 = x.to(at::kFloat);
    auto norm = x_fp32 * at::rsqrt(x_fp32.pow(2).mean(-1, true) + static_cast<double>(epsilon)) * weight.to(at::kFloat);
    auto norm_bf16 = norm.to(x.scalar_type());
    constexpr int64_t rope_dim = 64;
    auto rope = norm_bf16.slice(-1, norm_bf16.size(-1) - rope_dim, norm_bf16.size(-1)).to(at::kFloat).reshape({x.size(0), rope_dim / 2, 2});
    auto selected = freqs.index_select(0, pos).to(at::kFloat).reshape({x.size(0), rope_dim / 2, 2});
    auto c = selected.select(-1, 0);
    auto s = selected.select(-1, 1);
    auto xr = rope.select(-1, 0);
    auto xi = rope.select(-1, 1);
    auto rotated = at::stack({xr * c - xi * s, xr * s + xi * c}, -1).reshape({x.size(0), rope_dim});
    norm_bf16.slice(-1, norm_bf16.size(-1) - rope_dim, norm_bf16.size(-1)).copy_(rotated);
    x.copy_(norm_bf16);
#else
    (void)input;
    (void)norm_weight;
    (void)epsilon;
    (void)freqs_cis;
    (void)positions;
    throw std::runtime_error("deepseek_v4_fused_norm_rope_inplace_naive_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
