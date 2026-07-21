#include "infinicore/ops/deepseek_v4_moe_marlin_w8a8.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/SymInt.h>
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

void call_aiter_marlin(const char *schema_name,
                       const Tensor &input,
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
    check_accelerator_tensor(input, schema_name);
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
    auto topk_weights_at = to_optional_aten_tensor(topk_weights);
    auto sorted_token_ids_at = infinicore::adaptor::to_aten_tensor(sorted_token_ids);
    auto expert_ids_at = infinicore::adaptor::to_aten_tensor(expert_ids);
    auto num_tokens_post_pad_at = infinicore::adaptor::to_aten_tensor(num_tokens_post_pad);

    auto op = c10::Dispatcher::singleton()
                  .findSchemaOrThrow(schema_name, "")
                  .typed<at::Tensor(at::Tensor,
                                    at::Tensor,
                                    at::Tensor,
                                    at::Tensor,
                                    at::Tensor,
                                    std::optional<at::Tensor>,
                                    at::Tensor,
                                    at::Tensor,
                                    at::Tensor,
                                    c10::SymInt,
                                    c10::SymInt,
                                    c10::SymInt)>();
    op.call(input_at,
            b_qweight_at,
            output_at,
            a_scale_at,
            b_scale_at,
            topk_weights_at,
            sorted_token_ids_at,
            expert_ids_at,
            num_tokens_post_pad_at,
            c10::SymInt(top_k),
            c10::SymInt(mode),
            c10::SymInt(delta));
#else
    (void)schema_name;
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
    (void)mode;
    (void)delta;
    throw std::runtime_error("deepseek_v4_moe_marlin_w8a8 requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace

void deepseek_v4_moe_marlin_w8a8_(const Tensor &input,
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
    call_aiter_marlin("aiter::moe_c_moe_gemm_marlin_w8a8",
                      input,
                      b_qweight,
                      output,
                      a_scale,
                      b_scale,
                      topk_weights,
                      sorted_token_ids,
                      expert_ids,
                      num_tokens_post_pad,
                      top_k,
                      mode,
                      delta);
}

void deepseek_v4_moe_marlin_w8a8_fp8_(const Tensor &input,
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
    call_aiter_marlin("aiter::moe_c_moe_gemm_marlin_w8a8_fp8",
                      input,
                      b_qweight,
                      output,
                      a_scale,
                      b_scale,
                      topk_weights,
                      sorted_token_ids,
                      expert_ids,
                      num_tokens_post_pad,
                      top_k,
                      mode,
                      delta);
}

} // namespace infinicore::op
