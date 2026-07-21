#include "infinicore/ops/deepseek_v4_moe_align_block_size.hpp"

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

void deepseek_v4_moe_align_block_size_(const Tensor &topk_ids,
                                       int num_experts,
                                       int block_size,
                                       Tensor sorted_token_ids,
                                       Tensor experts_ids,
                                       Tensor num_tokens_post_pad,
                                       Tensor cumsum_buffer,
                                       bool pad_sorted_token_ids) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (topk_ids->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_moe_align_block_size_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (topk_ids->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_moe_align_block_size_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto topk_ids_at = infinicore::adaptor::to_aten_tensor(topk_ids);
    auto sorted_token_ids_at = infinicore::adaptor::to_aten_tensor(sorted_token_ids);
    auto experts_ids_at = infinicore::adaptor::to_aten_tensor(experts_ids);
    auto num_tokens_post_pad_at = infinicore::adaptor::to_aten_tensor(num_tokens_post_pad);
    auto cumsum_buffer_at = infinicore::adaptor::to_aten_tensor(cumsum_buffer);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::moe_align_block_size", "")
                         .typed<void(at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, bool)>();
    op.call(topk_ids_at,
            static_cast<int64_t>(num_experts),
            static_cast<int64_t>(block_size),
            sorted_token_ids_at,
            experts_ids_at,
            num_tokens_post_pad_at,
            cumsum_buffer_at,
            pad_sorted_token_ids);
#else
    (void)topk_ids;
    (void)num_experts;
    (void)block_size;
    (void)sorted_token_ids;
    (void)experts_ids;
    (void)num_tokens_post_pad;
    (void)cumsum_buffer;
    (void)pad_sorted_token_ids;
    throw std::runtime_error("deepseek_v4_moe_align_block_size_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
