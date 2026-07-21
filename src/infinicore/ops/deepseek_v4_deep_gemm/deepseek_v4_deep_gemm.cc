#include "infinicore/ops/deepseek_v4_deep_gemm.hpp"

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

namespace {

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
std::optional<at::Tensor> to_optional_aten_tensor(const std::optional<Tensor> &tensor) {
    if (!tensor.has_value()) {
        return std::nullopt;
    }
    return infinicore::adaptor::to_aten_tensor(*tensor);
}
#endif

void check_accelerator_tensor(const Tensor &tensor, const char *op_name) {
#if defined(ENABLE_HYGON_API)
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
#elif defined(ENABLE_NVIDIA_API)
    if (tensor->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error(std::string(op_name) + " expects NVIDIA tensors in this build.");
    }
#else
    (void)tensor;
    (void)op_name;
#endif
}

} // namespace

void deepseek_v4_deep_gemm_low_latency_grouped_gemm_(const Tensor &matrix_a,
                                                     const Tensor &matrix_b,
                                                     const Tensor &matrix_a_scale,
                                                     const Tensor &matrix_b_scale,
                                                     const Tensor &actual_tokens,
                                                     Tensor matrix_c,
                                                     int max_tokens,
                                                     int experts,
                                                     int cu_s,
                                                     bool block_wise,
                                                     bool b_overlap,
                                                     const std::optional<Tensor> &signal) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(matrix_a, "deepseek_v4_deep_gemm_low_latency_grouped_gemm_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto matrix_a_at = infinicore::adaptor::to_aten_tensor(matrix_a);
    auto matrix_b_at = infinicore::adaptor::to_aten_tensor(matrix_b);
    auto matrix_a_scale_at = infinicore::adaptor::to_aten_tensor(matrix_a_scale);
    auto matrix_b_scale_at = infinicore::adaptor::to_aten_tensor(matrix_b_scale);
    auto actual_tokens_at = infinicore::adaptor::to_aten_tensor(actual_tokens);
    auto matrix_c_at = infinicore::adaptor::to_aten_tensor(matrix_c);
    auto signal_at = to_optional_aten_tensor(signal);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("deep_gemm::low_latency_grouped_gemm", "")
                         .typed<at::Tensor(const at::Tensor &,
                                           const at::Tensor &,
                                           const at::Tensor &,
                                           const at::Tensor &,
                                           const at::Tensor &,
                                           at::Tensor &,
                                           int64_t,
                                           int64_t,
                                           int64_t,
                                           bool,
                                           bool,
                                           const std::optional<at::Tensor> &)>();
    op.call(matrix_a_at,
            matrix_b_at,
            matrix_a_scale_at,
            matrix_b_scale_at,
            actual_tokens_at,
            matrix_c_at,
            static_cast<int64_t>(max_tokens),
            static_cast<int64_t>(experts),
            static_cast<int64_t>(cu_s),
            block_wise,
            b_overlap,
            signal_at);
#else
    (void)matrix_a;
    (void)matrix_b;
    (void)matrix_a_scale;
    (void)matrix_b_scale;
    (void)actual_tokens;
    (void)matrix_c;
    (void)max_tokens;
    (void)experts;
    (void)cu_s;
    (void)block_wise;
    (void)b_overlap;
    (void)signal;
    throw std::runtime_error("deepseek_v4_deep_gemm_low_latency_grouped_gemm_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_deep_gemm_moe_w8a8_i8_marlin_prefill_down_(const Tensor &input,
                                                            const Tensor &b_qweight,
                                                            Tensor output,
                                                            const Tensor &a_scale,
                                                            const Tensor &b_scale,
                                                            const Tensor &topk_weights,
                                                            const Tensor &sorted_token_ids,
                                                            const Tensor &expert_ids,
                                                            const Tensor &num_tokens_post_pad,
                                                            int top_k,
                                                            int real_topk) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(input, "deepseek_v4_deep_gemm_moe_w8a8_i8_marlin_prefill_down_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto input_at = infinicore::adaptor::to_aten_tensor(input);
    auto b_qweight_at = infinicore::adaptor::to_aten_tensor(b_qweight);
    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    auto a_scale_at = infinicore::adaptor::to_aten_tensor(a_scale);
    auto b_scale_at = infinicore::adaptor::to_aten_tensor(b_scale);
    auto topk_weights_at = infinicore::adaptor::to_aten_tensor(topk_weights);
    auto sorted_token_ids_at = infinicore::adaptor::to_aten_tensor(sorted_token_ids);
    auto expert_ids_at = infinicore::adaptor::to_aten_tensor(expert_ids);
    auto num_tokens_post_pad_at = infinicore::adaptor::to_aten_tensor(num_tokens_post_pad);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("deep_gemm::moe_w8a8_i8_marlin_prefill_down", "")
                         .typed<at::Tensor(at::Tensor,
                                           at::Tensor,
                                           at::Tensor,
                                           at::Tensor,
                                           at::Tensor,
                                           at::Tensor,
                                           at::Tensor,
                                           at::Tensor,
                                           at::Tensor,
                                           int64_t,
                                           int64_t)>();
    op.call(input_at,
            b_qweight_at,
            output_at,
            a_scale_at,
            b_scale_at,
            topk_weights_at,
            sorted_token_ids_at,
            expert_ids_at,
            num_tokens_post_pad_at,
            static_cast<int64_t>(top_k),
            static_cast<int64_t>(real_topk));
#else
    (void)input;
    (void)b_qweight;
    (void)output;
    (void)a_scale;
    (void)b_scale;
    (void)topk_weights;
    (void)sorted_token_ids;
    (void)expert_ids;
    (void)num_tokens_post_pad;
    (void)top_k;
    (void)real_topk;
    throw std::runtime_error("deepseek_v4_deep_gemm_moe_w8a8_i8_marlin_prefill_down_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_deep_gemm_moe_w8a8_marlin_decode_down_fp8_(const Tensor &input,
                                                            const Tensor &b_qweight,
                                                            Tensor output,
                                                            const Tensor &a_scale,
                                                            const Tensor &b_scale,
                                                            const Tensor &topk_weights,
                                                            const Tensor &sorted_token_ids,
                                                            const Tensor &expert_ids,
                                                            const Tensor &num_tokens_post_pad,
                                                            int top_k,
                                                            int real_topk) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(input, "deepseek_v4_deep_gemm_moe_w8a8_marlin_decode_down_fp8_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto input_at = infinicore::adaptor::to_aten_tensor(input);
    auto b_qweight_at = infinicore::adaptor::to_aten_tensor(b_qweight);
    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    auto a_scale_at = infinicore::adaptor::to_aten_tensor(a_scale);
    auto b_scale_at = infinicore::adaptor::to_aten_tensor(b_scale);
    auto topk_weights_at = infinicore::adaptor::to_aten_tensor(topk_weights);
    auto sorted_token_ids_at = infinicore::adaptor::to_aten_tensor(sorted_token_ids);
    auto expert_ids_at = infinicore::adaptor::to_aten_tensor(expert_ids);
    auto num_tokens_post_pad_at = infinicore::adaptor::to_aten_tensor(num_tokens_post_pad);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("deep_gemm::moe_w8a8_marlin_decode_down_fp8", "")
                         .typed<at::Tensor(at::Tensor,
                                           at::Tensor,
                                           at::Tensor,
                                           at::Tensor,
                                           at::Tensor,
                                           at::Tensor,
                                           at::Tensor,
                                           at::Tensor,
                                           at::Tensor,
                                           int64_t,
                                           int64_t)>();
    op.call(input_at,
            b_qweight_at,
            output_at,
            a_scale_at,
            b_scale_at,
            topk_weights_at,
            sorted_token_ids_at,
            expert_ids_at,
            num_tokens_post_pad_at,
            static_cast<int64_t>(top_k),
            static_cast<int64_t>(real_topk));
#else
    (void)input;
    (void)b_qweight;
    (void)output;
    (void)a_scale;
    (void)b_scale;
    (void)topk_weights;
    (void)sorted_token_ids;
    (void)expert_ids;
    (void)num_tokens_post_pad;
    (void)top_k;
    (void)real_topk;
    throw std::runtime_error("deepseek_v4_deep_gemm_moe_w8a8_marlin_decode_down_fp8_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
