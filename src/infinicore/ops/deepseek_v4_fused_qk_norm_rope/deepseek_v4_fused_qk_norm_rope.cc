#include "infinicore/ops/deepseek_v4_fused_qk_norm_rope.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/core/dispatch/Dispatcher.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <stdexcept>

namespace infinicore::op {

void deepseek_v4_fused_qk_norm_rope_(Tensor qkv,
                                     int num_heads_q,
                                     int num_heads_k,
                                     int num_heads_v,
                                     int head_dim,
                                     float eps,
                                     const Tensor &q_weight,
                                     const Tensor &k_weight,
                                     const Tensor &cos_sin_cache,
                                     bool is_neox,
                                     const Tensor &position_ids) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (qkv->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_fused_qk_norm_rope_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (qkv->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_fused_qk_norm_rope_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto qkv_at = infinicore::adaptor::to_aten_tensor(qkv);
    auto q_weight_at = infinicore::adaptor::to_aten_tensor(q_weight);
    auto k_weight_at = infinicore::adaptor::to_aten_tensor(k_weight);
    auto cos_sin_cache_at = infinicore::adaptor::to_aten_tensor(cos_sin_cache);
    auto position_ids_at = infinicore::adaptor::to_aten_tensor(position_ids);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("_C::fused_qk_norm_rope", "")
                         .typed<void(at::Tensor &, int64_t, int64_t, int64_t, int64_t, double, at::Tensor &, at::Tensor &, at::Tensor &, bool, at::Tensor &)>();
    op.call(qkv_at,
            static_cast<int64_t>(num_heads_q),
            static_cast<int64_t>(num_heads_k),
            static_cast<int64_t>(num_heads_v),
            static_cast<int64_t>(head_dim),
            static_cast<double>(eps),
            q_weight_at,
            k_weight_at,
            cos_sin_cache_at,
            is_neox,
            position_ids_at);
#else
    (void)qkv;
    (void)num_heads_q;
    (void)num_heads_k;
    (void)num_heads_v;
    (void)head_dim;
    (void)eps;
    (void)q_weight;
    (void)k_weight;
    (void)cos_sin_cache;
    (void)is_neox;
    (void)position_ids;
    throw std::runtime_error("deepseek_v4_fused_qk_norm_rope_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
