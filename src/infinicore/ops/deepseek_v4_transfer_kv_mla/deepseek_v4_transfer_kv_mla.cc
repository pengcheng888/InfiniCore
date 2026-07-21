#include "infinicore/ops/deepseek_v4_transfer_kv_mla.hpp"

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

void deepseek_v4_transfer_kv_per_layer_mla_(const Tensor &src,
                                            Tensor dst,
                                            const Tensor &src_indices,
                                            const Tensor &dst_indices,
                                            int item_size,
                                            int block_quota,
                                            int num_warps_per_block) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (src->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_transfer_kv_per_layer_mla_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (src->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_transfer_kv_per_layer_mla_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto src_at = infinicore::adaptor::to_aten_tensor(src);
    auto dst_at = infinicore::adaptor::to_aten_tensor(dst);
    auto src_indices_at = infinicore::adaptor::to_aten_tensor(src_indices);
    auto dst_indices_at = infinicore::adaptor::to_aten_tensor(dst_indices);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::transfer_kv_per_layer_mla", "")
                         .typed<void(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t)>();
    op.call(src_at,
            dst_at,
            src_indices_at,
            dst_indices_at,
            static_cast<int64_t>(item_size),
            static_cast<int64_t>(block_quota),
            static_cast<int64_t>(num_warps_per_block));
#else
    (void)src;
    (void)dst;
    (void)src_indices;
    (void)dst_indices;
    (void)item_size;
    (void)block_quota;
    (void)num_warps_per_block;
    throw std::runtime_error("deepseek_v4_transfer_kv_per_layer_mla_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_transfer_kv_per_layer_mla_pf_lf_(const Tensor &src,
                                                  Tensor dst,
                                                  const Tensor &src_indices,
                                                  const Tensor &dst_indices,
                                                  int layer_id,
                                                  int item_size,
                                                  int src_layout_dim,
                                                  int block_quota,
                                                  int num_warps_per_block) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (src->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_transfer_kv_per_layer_mla_pf_lf_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (src->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_transfer_kv_per_layer_mla_pf_lf_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto src_at = infinicore::adaptor::to_aten_tensor(src);
    auto dst_at = infinicore::adaptor::to_aten_tensor(dst);
    auto src_indices_at = infinicore::adaptor::to_aten_tensor(src_indices);
    auto dst_indices_at = infinicore::adaptor::to_aten_tensor(dst_indices);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::transfer_kv_per_layer_mla_pf_lf", "")
                         .typed<void(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t)>();
    op.call(src_at,
            dst_at,
            src_indices_at,
            dst_indices_at,
            static_cast<int64_t>(layer_id),
            static_cast<int64_t>(item_size),
            static_cast<int64_t>(src_layout_dim),
            static_cast<int64_t>(block_quota),
            static_cast<int64_t>(num_warps_per_block));
#else
    (void)src;
    (void)dst;
    (void)src_indices;
    (void)dst_indices;
    (void)layer_id;
    (void)item_size;
    (void)src_layout_dim;
    (void)block_quota;
    (void)num_warps_per_block;
    throw std::runtime_error("deepseek_v4_transfer_kv_per_layer_mla_pf_lf_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
