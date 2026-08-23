#include "infinicore/ops/deepseek_v4_create_chunked_prefix_cache_kv_indices.hpp"

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

void deepseek_v4_create_chunked_prefix_cache_kv_indices_(const Tensor &req_to_token,
                                                         const Tensor &req_pool_indices,
                                                         const Tensor &chunk_starts,
                                                         const Tensor &chunk_seq_lens,
                                                         const Tensor &chunk_cu_seq_lens,
                                                         Tensor chunk_kv_indices,
                                                         int col_num,
                                                         int bs) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (req_to_token->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_create_chunked_prefix_cache_kv_indices_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (req_to_token->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_create_chunked_prefix_cache_kv_indices_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto req_to_token_at = infinicore::adaptor::to_aten_tensor(req_to_token);
    auto req_pool_indices_at = infinicore::adaptor::to_aten_tensor(req_pool_indices);
    auto chunk_starts_at = infinicore::adaptor::to_aten_tensor(chunk_starts);
    auto chunk_seq_lens_at = infinicore::adaptor::to_aten_tensor(chunk_seq_lens);
    auto chunk_cu_seq_lens_at = infinicore::adaptor::to_aten_tensor(chunk_cu_seq_lens);
    auto chunk_kv_indices_at = infinicore::adaptor::to_aten_tensor(chunk_kv_indices);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::dcu_create_chunked_prefix_cache_kv_indices", "")
                         .typed<void(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t)>();
    op.call(req_to_token_at,
            req_pool_indices_at,
            chunk_starts_at,
            chunk_seq_lens_at,
            chunk_cu_seq_lens_at,
            chunk_kv_indices_at,
            static_cast<int64_t>(col_num),
            static_cast<int64_t>(bs));
#else
    (void)req_to_token;
    (void)req_pool_indices;
    (void)chunk_starts;
    (void)chunk_seq_lens;
    (void)chunk_cu_seq_lens;
    (void)chunk_kv_indices;
    (void)col_num;
    (void)bs;
    throw std::runtime_error("deepseek_v4_create_chunked_prefix_cache_kv_indices_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
