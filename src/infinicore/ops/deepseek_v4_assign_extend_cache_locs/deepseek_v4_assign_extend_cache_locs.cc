#include "infinicore/ops/deepseek_v4_assign_extend_cache_locs.hpp"

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

void deepseek_v4_assign_extend_cache_locs_(const Tensor &req_pool_indices,
                                           const Tensor &req_to_token,
                                           const Tensor &start_offset,
                                           const Tensor &end_offset,
                                           Tensor out_cache_loc,
                                           int pool_len,
                                           int bs) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (req_pool_indices->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_assign_extend_cache_locs_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (req_pool_indices->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_assign_extend_cache_locs_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto req_pool_indices_at = infinicore::adaptor::to_aten_tensor(req_pool_indices);
    auto req_to_token_at = infinicore::adaptor::to_aten_tensor(req_to_token);
    auto start_offset_at = infinicore::adaptor::to_aten_tensor(start_offset);
    auto end_offset_at = infinicore::adaptor::to_aten_tensor(end_offset);
    auto out_cache_loc_at = infinicore::adaptor::to_aten_tensor(out_cache_loc);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::dcu_assign_extend_cache_locs", "")
                         .typed<void(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t)>();
    op.call(req_pool_indices_at,
            req_to_token_at,
            start_offset_at,
            end_offset_at,
            out_cache_loc_at,
            static_cast<int64_t>(pool_len),
            static_cast<int64_t>(bs));
#else
    (void)req_pool_indices;
    (void)req_to_token;
    (void)start_offset;
    (void)end_offset;
    (void)out_cache_loc;
    (void)pool_len;
    (void)bs;
    throw std::runtime_error("deepseek_v4_assign_extend_cache_locs_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
