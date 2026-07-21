#include "infinicore/ops/deepseek_v4_create_flashmla_kv_indices.hpp"

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

void deepseek_v4_create_flashmla_kv_indices_(const Tensor &req_to_token,
                                             const Tensor &req_pool_indices,
                                             const Tensor &page_kernel_lens,
                                             std::optional<Tensor> kv_start_idx,
                                             Tensor kv_indices,
                                             int req_to_token_stride,
                                             int kv_indices_stride,
                                             int page_size) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (req_to_token->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_create_flashmla_kv_indices_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (req_to_token->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_create_flashmla_kv_indices_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto req_to_token_at = infinicore::adaptor::to_aten_tensor(req_to_token);
    auto req_pool_indices_at = infinicore::adaptor::to_aten_tensor(req_pool_indices);
    auto page_kernel_lens_at = infinicore::adaptor::to_aten_tensor(page_kernel_lens);
    auto kv_indices_at = infinicore::adaptor::to_aten_tensor(kv_indices);
    std::optional<at::Tensor> kv_start_idx_at = std::nullopt;
    if (kv_start_idx.has_value()) {
        kv_start_idx_at = infinicore::adaptor::to_aten_tensor(*kv_start_idx);
    }

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::dcu_create_flashmla_kv_indices", "")
                         .typed<void(const at::Tensor &, const at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &, at::Tensor &, int64_t, int64_t, int64_t)>();
    op.call(req_to_token_at,
            req_pool_indices_at,
            page_kernel_lens_at,
            kv_start_idx_at,
            kv_indices_at,
            static_cast<int64_t>(req_to_token_stride),
            static_cast<int64_t>(kv_indices_stride),
            static_cast<int64_t>(page_size));
#else
    (void)req_to_token;
    (void)req_pool_indices;
    (void)page_kernel_lens;
    (void)kv_start_idx;
    (void)kv_indices;
    (void)req_to_token_stride;
    (void)kv_indices_stride;
    (void)page_size;
    throw std::runtime_error("deepseek_v4_create_flashmla_kv_indices_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
