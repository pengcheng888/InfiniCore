#include "infinicore/ops/deepseek_v4_fast_topk.hpp"

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

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
namespace {
void guard_deepseek_v4_fast_topk_device(const Tensor &tensor, const char *op_name) {
#if defined(ENABLE_HYGON_API)
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
#else
    if (tensor->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error(std::string(op_name) + " expects NVIDIA tensors in this build.");
    }
#endif
}

std::optional<at::Tensor> to_optional_aten(std::optional<Tensor> tensor) {
    if (!tensor.has_value()) {
        return std::nullopt;
    }
    return infinicore::adaptor::to_aten_tensor(*tensor);
}
} // namespace
#endif

void deepseek_v4_fast_topk_(const Tensor &score,
                            Tensor indices,
                            const Tensor &lengths,
                            std::optional<Tensor> row_starts) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    guard_deepseek_v4_fast_topk_device(score, "deepseek_v4_fast_topk_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto score_at = infinicore::adaptor::to_aten_tensor(score);
    auto indices_at = infinicore::adaptor::to_aten_tensor(indices);
    auto lengths_at = infinicore::adaptor::to_aten_tensor(lengths);
    auto row_starts_at = to_optional_aten(row_starts);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::fast_topk", "")
                         .typed<void(const at::Tensor &, at::Tensor &, const at::Tensor &, std::optional<at::Tensor>)>();
    op.call(score_at, indices_at, lengths_at, row_starts_at);
#else
    (void)score;
    (void)indices;
    (void)lengths;
    (void)row_starts;
    throw std::runtime_error("deepseek_v4_fast_topk_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_fast_topk_transform_fused_(const Tensor &score,
                                            const Tensor &lengths,
                                            Tensor dst_page_table,
                                            const Tensor &src_page_table,
                                            const Tensor &cu_seqlens_q,
                                            std::optional<Tensor> row_starts) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    guard_deepseek_v4_fast_topk_device(score, "deepseek_v4_fast_topk_transform_fused_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto score_at = infinicore::adaptor::to_aten_tensor(score);
    auto lengths_at = infinicore::adaptor::to_aten_tensor(lengths);
    auto dst_page_table_at = infinicore::adaptor::to_aten_tensor(dst_page_table);
    auto src_page_table_at = infinicore::adaptor::to_aten_tensor(src_page_table);
    auto cu_seqlens_q_at = infinicore::adaptor::to_aten_tensor(cu_seqlens_q);
    auto row_starts_at = to_optional_aten(row_starts);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::fast_topk_transform_fused", "")
                         .typed<void(const at::Tensor &, const at::Tensor &, at::Tensor &, const at::Tensor &, const at::Tensor &, std::optional<at::Tensor>)>();
    op.call(score_at, lengths_at, dst_page_table_at, src_page_table_at, cu_seqlens_q_at, row_starts_at);
#else
    (void)score;
    (void)lengths;
    (void)dst_page_table;
    (void)src_page_table;
    (void)cu_seqlens_q;
    (void)row_starts;
    throw std::runtime_error("deepseek_v4_fast_topk_transform_fused_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_fast_topk_transform_ragged_fused_(const Tensor &score,
                                                   const Tensor &lengths,
                                                   Tensor topk_indices_ragged,
                                                   const Tensor &topk_indices_offset,
                                                   std::optional<Tensor> row_starts) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    guard_deepseek_v4_fast_topk_device(score, "deepseek_v4_fast_topk_transform_ragged_fused_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto score_at = infinicore::adaptor::to_aten_tensor(score);
    auto lengths_at = infinicore::adaptor::to_aten_tensor(lengths);
    auto topk_indices_ragged_at = infinicore::adaptor::to_aten_tensor(topk_indices_ragged);
    auto topk_indices_offset_at = infinicore::adaptor::to_aten_tensor(topk_indices_offset);
    auto row_starts_at = to_optional_aten(row_starts);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::fast_topk_transform_ragged_fused", "")
                         .typed<void(const at::Tensor &, const at::Tensor &, at::Tensor &, const at::Tensor &, std::optional<at::Tensor>)>();
    op.call(score_at, lengths_at, topk_indices_ragged_at, topk_indices_offset_at, row_starts_at);
#else
    (void)score;
    (void)lengths;
    (void)topk_indices_ragged;
    (void)topk_indices_offset;
    (void)row_starts;
    throw std::runtime_error("deepseek_v4_fast_topk_transform_ragged_fused_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
