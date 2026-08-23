#include "infinicore/ops/deepseek_v4_compress_fused_norm_rope.hpp"

#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace {

void check_hygon_or_nvidia_tensor(const Tensor &tensor, const char *op_name) {
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

void check_compress_fused_norm_rope_shapes(const Tensor &input,
                                           const Tensor &norm_weight,
                                           const Tensor &freqs_cis,
                                           const Tensor &positions) {
    if (input->ndim() != 2 || input->size(1) < 64) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_naive_ expects input [tokens, dim>=64].");
    }
    if (input->dtype() != DataType::BF16) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_naive_ expects bf16 input.");
    }
    if (norm_weight->numel() != input->size(1)) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_naive_ norm_weight size mismatch.");
    }
    if (freqs_cis->ndim() != 2 || freqs_cis->size(1) != 64 || freqs_cis->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_naive_ expects freqs_cis [max_pos, 64] float32.");
    }
    if (positions->ndim() != 1 || positions->numel() != input->size(0) ||
        (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64)) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_naive_ expects positions [tokens] int32/int64.");
    }
}

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
void apply_rope_2d_last64_aten_(at::Tensor rope, const at::Tensor &freqs_cis, const at::Tensor &positions) {
    constexpr int64_t rope_dim = 64;
    const int64_t tokens = rope.size(0);
    if (tokens == 0) {
        return;
    }
    auto pos_long = positions.reshape({tokens}).to(at::kLong);
    auto selected = freqs_cis.index_select(0, pos_long).to(at::kFloat).reshape({tokens, rope_dim / 2, 2});
    auto freq_real = selected.select(-1, 0);
    auto freq_imag = selected.select(-1, 1);

    auto rope_pair = rope.to(at::kFloat).reshape({tokens, rope_dim / 2, 2});
    auto x_real = rope_pair.select(-1, 0);
    auto x_imag = rope_pair.select(-1, 1);
    auto out_real = x_real * freq_real - x_imag * freq_imag;
    auto out_imag = x_real * freq_imag + x_imag * freq_real;
    auto result = at::stack({out_real, out_imag}, -1).reshape(rope.sizes()).to(rope.scalar_type());
    rope.copy_(result);
}
#endif

} // namespace

void deepseek_v4_compress_fused_norm_rope_naive_(Tensor input,
                                                     const Tensor &norm_weight,
                                                     float epsilon,
                                                     const Tensor &freqs_cis,
                                                     const Tensor &positions) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_hygon_or_nvidia_tensor(input, "deepseek_v4_compress_fused_norm_rope_naive_");
    check_hygon_or_nvidia_tensor(norm_weight, "deepseek_v4_compress_fused_norm_rope_naive_");
    check_hygon_or_nvidia_tensor(freqs_cis, "deepseek_v4_compress_fused_norm_rope_naive_");
    check_hygon_or_nvidia_tensor(positions, "deepseek_v4_compress_fused_norm_rope_naive_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    check_compress_fused_norm_rope_shapes(input, norm_weight, freqs_cis, positions);

    auto input_at = infinicore::adaptor::to_aten_tensor(input);
    if (!input_at.is_contiguous()) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_naive_ expects contiguous input.");
    }
    const int64_t input_dim = input_at.size(1);
    auto weight_at = infinicore::adaptor::to_aten_tensor(norm_weight).to(at::kFloat).reshape({1, input_dim});
    auto input_float = input_at.to(at::kFloat);
    auto variance = (input_float * input_float).mean({-1}, true);
    auto normalized = input_float * at::rsqrt(variance + static_cast<double>(epsilon)) * weight_at;
    input_at.copy_(normalized.to(input_at.scalar_type()));

    auto rope = input_at.slice(1, input_dim - 64, input_dim);
    apply_rope_2d_last64_aten_(rope,
                               infinicore::adaptor::to_aten_tensor(freqs_cis),
                               infinicore::adaptor::to_aten_tensor(positions));
#else
    (void)input;
    (void)norm_weight;
    (void)epsilon;
    (void)freqs_cis;
    (void)positions;
    throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_naive_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}




} // namespace infinicore::op
