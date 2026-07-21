#include "infinicore/ops/deepseek_v4_assign_req_to_token_pool.hpp"

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

void deepseek_v4_assign_req_to_token_pool_(const Tensor &req_pool_indices,
                                           Tensor req_to_token,
                                           const Tensor &allocate_lens,
                                           const Tensor &new_allocate_lens,
                                           Tensor out_cache_loc,
                                           int shape,
                                           int bs) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (req_pool_indices->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_assign_req_to_token_pool_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (req_pool_indices->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_assign_req_to_token_pool_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto req_pool_indices_at = infinicore::adaptor::to_aten_tensor(req_pool_indices);
    auto req_to_token_at = infinicore::adaptor::to_aten_tensor(req_to_token);
    auto allocate_lens_at = infinicore::adaptor::to_aten_tensor(allocate_lens);
    auto new_allocate_lens_at = infinicore::adaptor::to_aten_tensor(new_allocate_lens);
    auto out_cache_loc_at = infinicore::adaptor::to_aten_tensor(out_cache_loc);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::dcu_assign_req_to_token_pool", "")
                         .typed<void(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t)>();
    op.call(req_pool_indices_at,
            req_to_token_at,
            allocate_lens_at,
            new_allocate_lens_at,
            out_cache_loc_at,
            static_cast<int64_t>(shape),
            static_cast<int64_t>(bs));
#else
    (void)req_pool_indices;
    (void)req_to_token;
    (void)allocate_lens;
    (void)new_allocate_lens;
    (void)out_cache_loc;
    (void)shape;
    (void)bs;
    throw std::runtime_error("deepseek_v4_assign_req_to_token_pool_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
