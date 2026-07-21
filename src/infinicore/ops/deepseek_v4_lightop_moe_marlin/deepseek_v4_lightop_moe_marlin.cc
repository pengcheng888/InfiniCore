#include "infinicore/ops/deepseek_v4_lightop_moe_marlin.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
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
void guard_device(const Tensor &tensor, const char *op_name) {
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

void push_optional_tensor(c10::Stack &stack, const std::optional<Tensor> &tensor) {
    if (tensor.has_value()) {
        stack.emplace_back(infinicore::adaptor::to_aten_tensor(*tensor));
    } else {
        stack.emplace_back();
    }
}
#endif

} // namespace

void deepseek_v4_lightop_moe_gemm_marlin_w8a8_(const Tensor &input,
                                                const Tensor &b_qweight,
                                                Tensor output,
                                                const Tensor &a_scale,
                                                const Tensor &b_scale,
                                                const std::optional<Tensor> &topk_weights,
                                                const Tensor &sorted_token_ids,
                                                const Tensor &expert_ids,
                                                const Tensor &num_tokens_post_pad,
                                                int top_k,
                                                int mode,
                                                int delta) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    guard_device(input, "deepseek_v4_lightop_moe_gemm_marlin_w8a8_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    c10::Stack stack;
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(input));
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(b_qweight));
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(output));
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(a_scale));
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(b_scale));
    push_optional_tensor(stack, topk_weights);
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(sorted_token_ids));
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(expert_ids));
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(num_tokens_post_pad));
    stack.emplace_back(static_cast<int64_t>(top_k));
    stack.emplace_back(static_cast<int64_t>(mode));
    stack.emplace_back(static_cast<int64_t>(delta));
    auto op = c10::Dispatcher::singleton().findSchemaOrThrow("infinicore_deepseek_v4::lightop_moe_gemm_marlin_w8a8", "");
    op.callBoxed(&stack);
#else
    (void)input; (void)b_qweight; (void)output; (void)a_scale; (void)b_scale; (void)topk_weights;
    (void)sorted_token_ids; (void)expert_ids; (void)num_tokens_post_pad; (void)top_k; (void)mode; (void)delta;
    throw std::runtime_error("deepseek_v4_lightop_moe_gemm_marlin_w8a8_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_lightop_fuse_silu_mul_quant_(Tensor output,
                                               Tensor scales,
                                               const Tensor &input,
                                               const std::optional<Tensor> &num_local_tokens_tensor,
                                               int topk,
                                               int expect_m,
                                               const std::optional<Tensor> &expert_ids) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    guard_device(input, "deepseek_v4_lightop_fuse_silu_mul_quant_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    c10::Stack stack;
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(input));
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(output));
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(scales));
    push_optional_tensor(stack, num_local_tokens_tensor);
    stack.emplace_back(static_cast<int64_t>(topk));
    stack.emplace_back(static_cast<int64_t>(expect_m));
    push_optional_tensor(stack, expert_ids);
    auto op = c10::Dispatcher::singleton().findSchemaOrThrow("infinicore_deepseek_v4::lightop_fuse_silu_mul_quant", "");
    op.callBoxed(&stack);
#else
    (void)output; (void)scales; (void)input; (void)num_local_tokens_tensor; (void)topk; (void)expect_m; (void)expert_ids;
    throw std::runtime_error("deepseek_v4_lightop_fuse_silu_mul_quant_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_lightop_moe_sum_(Tensor output,
                                   const Tensor &input,
                                   const std::optional<Tensor> &bias,
                                   const std::optional<Tensor> &expert_mask,
                                   const std::optional<Tensor> &num_local_tokens,
                                   float factor,
                                   int expect_m) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    guard_device(input, "deepseek_v4_lightop_moe_sum_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    c10::Stack stack;
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(input));
    stack.emplace_back(infinicore::adaptor::to_aten_tensor(output));
    push_optional_tensor(stack, bias);
    push_optional_tensor(stack, expert_mask);
    push_optional_tensor(stack, num_local_tokens);
    stack.emplace_back(static_cast<double>(factor));
    stack.emplace_back(static_cast<int64_t>(expect_m));
    auto op = c10::Dispatcher::singleton().findSchemaOrThrow("infinicore_deepseek_v4::lightop_moe_sum", "");
    op.callBoxed(&stack);
#else
    (void)output; (void)input; (void)bias; (void)expert_mask; (void)num_local_tokens; (void)factor; (void)expect_m;
    throw std::runtime_error("deepseek_v4_lightop_moe_sum_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
