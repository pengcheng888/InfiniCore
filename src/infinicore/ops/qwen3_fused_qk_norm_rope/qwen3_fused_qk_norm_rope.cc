#include "infinicore/ops/qwen3_fused_qk_norm_rope.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/core/dispatch/Dispatcher.h>
#if defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <stdexcept>

namespace infinicore::op {

void qwen3_fused_qk_norm_rope_(Tensor qkv,
                               int num_heads_q,
                               int num_heads_k,
                               int num_heads_v,
                               int head_dim,
                               float eps,
                               const Tensor &q_weight,
                               const Tensor &k_weight,
                               float base,
                               bool is_neox,
                               const Tensor &position_ids,
                               float factor,
                               float low,
                               float high,
                               float attention_factor,
                               int rotary_dim) {
#if defined(ENABLE_ATEN) && defined(ENABLE_NVIDIA_API)
    if (qkv->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("qwen3_fused_qk_norm_rope_ currently supports NVIDIA tensors only.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());

    auto qkv_at = infinicore::adaptor::to_aten_tensor(qkv);
    auto q_weight_at = infinicore::adaptor::to_aten_tensor(q_weight);
    auto k_weight_at = infinicore::adaptor::to_aten_tensor(k_weight);
    auto position_ids_at = infinicore::adaptor::to_aten_tensor(position_ids);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::fused_qk_norm_rope", "")
                         .typed<void(at::Tensor &, int64_t, int64_t, int64_t, int64_t, double, at::Tensor &, at::Tensor &, double, bool, at::Tensor &, double, double, double, double, int64_t)>();
    op.call(qkv_at,
            static_cast<int64_t>(num_heads_q),
            static_cast<int64_t>(num_heads_k),
            static_cast<int64_t>(num_heads_v),
            static_cast<int64_t>(head_dim),
            static_cast<double>(eps),
            q_weight_at,
            k_weight_at,
            static_cast<double>(base),
            is_neox,
            position_ids_at,
            static_cast<double>(factor),
            static_cast<double>(low),
            static_cast<double>(high),
            static_cast<double>(attention_factor),
            static_cast<int64_t>(rotary_dim));
#else
    (void)qkv;
    (void)num_heads_q;
    (void)num_heads_k;
    (void)num_heads_v;
    (void)head_dim;
    (void)eps;
    (void)q_weight;
    (void)k_weight;
    (void)base;
    (void)is_neox;
    (void)position_ids;
    (void)factor;
    (void)low;
    (void)high;
    (void)attention_factor;
    (void)rotary_dim;
    throw std::runtime_error("qwen3_fused_qk_norm_rope_ requires an ATen-enabled NVIDIA build.");
#endif
}

} // namespace infinicore::op
