#include "infinicore/ops/deepseek_v4_fused_q_norm_rope.hpp"

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

void deepseek_v4_fused_q_norm_rope_aten_(Tensor q_out,
                                         const Tensor &q_input,
                                         float epsilon,
                                         const Tensor &freqs_cis,
                                         const Tensor &positions) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    auto q = infinicore::adaptor::to_aten_tensor(q_input);
    auto out = infinicore::adaptor::to_aten_tensor(q_out);
    auto freqs = infinicore::adaptor::to_aten_tensor(freqs_cis);
    auto pos = infinicore::adaptor::to_aten_tensor(positions).to(at::kLong);
    auto q_fp32 = q.to(at::kFloat);
    auto norm = (q_fp32 * at::rsqrt(q_fp32.pow(2).mean(-1, true) + static_cast<double>(epsilon))).to(out.scalar_type());
    constexpr int64_t rope_dim = 64;
    auto tail = norm.slice(-1, norm.size(-1) - rope_dim, norm.size(-1));
    auto rope = tail.to(at::kFloat).reshape({q.size(0), q.size(1), rope_dim / 2, 2});
    auto selected = freqs.index_select(0, pos).to(at::kFloat).reshape({q.size(0), rope_dim / 2, 2});
    auto c = selected.select(-1, 0).unsqueeze(1);
    auto s = selected.select(-1, 1).unsqueeze(1);
    auto xr = rope.select(-1, 0);
    auto xi = rope.select(-1, 1);
    auto rotated = at::stack({xr * c - xi * s, xr * s + xi * c}, -1).reshape({q.size(0), q.size(1), rope_dim});
    tail.copy_(rotated.to(norm.scalar_type()));
    out.copy_(norm);
#else
    (void)q_out;
    (void)q_input;
    (void)epsilon;
    (void)freqs_cis;
    (void)positions;
    throw std::runtime_error("deepseek_v4_fused_q_norm_rope_aten_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
