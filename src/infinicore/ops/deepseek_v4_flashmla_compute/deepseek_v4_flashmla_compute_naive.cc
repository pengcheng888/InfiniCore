#include "infinicore/ops/deepseek_v4_flashmla_compute.hpp"

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

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op {

namespace {

constexpr int64_t kDsv4FlashMlaQDim = 512;

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



Tensor deepseek_v4_c4_compress_prefill_naive(const Tensor &kv_score_input,
                                                 const Tensor &ape) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    check_hygon_or_nvidia_tensor(kv_score_input, "deepseek_v4_c4_compress_prefill_naive");
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 4 != 0) {
        throw std::runtime_error("deepseek_v4_c4_compress_prefill_naive expects kv_score_input [tokens, 4 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 4);
    if (head_dim != kDsv4FlashMlaQDim) {
        throw std::runtime_error("deepseek_v4_c4_compress_prefill_naive expects head_dim 512.");
    }
    if (ape->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_c4_compress_prefill_naive expects ape rank 2.");
    }

    auto output = Tensor::zeros({static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}, kv_score_input->dtype(), kv_score_input->device());
    if (tokens == 0) {
        return output;
    }

    auto kv_score_at = infinicore::adaptor::to_aten_tensor(kv_score_input).contiguous().to(at::kFloat)
                           .reshape({tokens, 4, head_dim});
    auto ape_at = infinicore::adaptor::to_aten_tensor(ape).contiguous().to(at::kFloat);
    at::Tensor ape_view;
    if (ape_at.dim() == 2 && ape_at.size(0) == 4 && ape_at.size(1) == 2 * head_dim) {
        auto ape_chunks = ape_at.reshape({4, 2, head_dim});
        // SGLang applies the non-2604 C4 APE hotfix after loading: [score, overlap] -> [overlap, score].
        ape_view = at::cat({ape_chunks.select(1, 1), ape_chunks.select(1, 0)}, 0).contiguous();
    } else if (ape_at.dim() == 2 && ape_at.size(0) == 8 && ape_at.size(1) == head_dim) {
        ape_view = ape_at;
    } else {
        throw std::runtime_error("deepseek_v4_c4_compress_prefill_naive expects ape [4, 1024] or [8, 512].");
    }

    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    for (int64_t end = 3; end < tokens; end += 4) {
        std::vector<at::Tensor> kv_parts;
        std::vector<at::Tensor> score_parts;
        std::vector<at::Tensor> bias_parts;
        const int64_t overlap_start = std::max<int64_t>(0, end - 7);
        const int64_t overlap_end = end - 3;
        if (overlap_end > overlap_start) {
            const int64_t overlap_len = overlap_end - overlap_start;
            auto overlap = kv_score_at.slice(0, overlap_start, overlap_end);
            kv_parts.push_back(overlap.select(1, 0));
            score_parts.push_back(overlap.select(1, 2));
            bias_parts.push_back(ape_view.slice(0, 4 - overlap_len, 4));
        }

        const int64_t normal_start = std::max<int64_t>(0, end - 3);
        const int64_t normal_end = end + 1;
        auto normal = kv_score_at.slice(0, normal_start, normal_end);
        const int64_t normal_len = normal_end - normal_start;
        kv_parts.push_back(normal.select(1, 1));
        score_parts.push_back(normal.select(1, 3));
        bias_parts.push_back(ape_view.slice(0, 8 - normal_len, 8));

        auto kv_window = at::cat(kv_parts, 0);
        auto score_window = at::cat(score_parts, 0) + at::cat(bias_parts, 0);
        auto prob = at::softmax(score_window, 0);
        auto compressed = (kv_window * prob).sum(0);
        output_at.select(0, end).copy_(compressed.to(output_at.scalar_type()));
    }
    return output;
#else
    (void)kv_score_input;
    (void)ape;
    throw std::runtime_error("deepseek_v4_c4_compress_prefill_naive requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}




Tensor deepseek_v4_c4_compress_stateful_naive(const Tensor &kv_score_input,
                                                  const Tensor &ape,
                                                  Tensor compressor_state,
                                                  const Tensor &write_loc,
                                                  const Tensor &extra_loc,
                                                  const Tensor &positions) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    check_hygon_or_nvidia_tensor(kv_score_input, "deepseek_v4_c4_compress_stateful_naive");
    check_hygon_or_nvidia_tensor(compressor_state, "deepseek_v4_c4_compress_stateful_naive");
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 4 != 0) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_naive expects kv_score_input [tokens, 4 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 4);
    if (head_dim <= 0) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_naive expects positive head_dim.");
    }
    if (compressor_state->ndim() != 2 || compressor_state->size(1) != static_cast<size_t>(4 * head_dim) || compressor_state->size(0) % 4 != 0) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_naive expects compressor_state [4 * groups, 4 * head_dim].");
    }
    if (write_loc->numel() != static_cast<size_t>(tokens) || positions->numel() != static_cast<size_t>(tokens)) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_naive metadata token count mismatch.");
    }

    auto output = Tensor::zeros({static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}, kv_score_input->dtype(), kv_score_input->device());
    if (tokens == 0) {
        return output;
    }

    auto state_at = infinicore::adaptor::to_aten_tensor(compressor_state);
    if (!state_at.is_contiguous()) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_naive expects contiguous compressor_state.");
    }
    auto kv_score_at = infinicore::adaptor::to_aten_tensor(kv_score_input).contiguous().to(state_at.scalar_type())
                           .reshape({tokens, 4 * head_dim});
    auto state_groups = state_at.view({static_cast<int64_t>(compressor_state->size(0)) / 4, 4, 4, head_dim});

    auto write_loc_at = infinicore::adaptor::to_aten_tensor(write_loc).reshape({tokens}).to(at::kLong);
    at::Tensor extra_prev_at;
    auto extra_at_raw = infinicore::adaptor::to_aten_tensor(extra_loc).to(at::kLong);
    if (extra_at_raw.dim() == 2) {
        if (extra_at_raw.size(0) != tokens || extra_at_raw.size(1) < 1) {
            throw std::runtime_error("deepseek_v4_c4_compress_stateful_naive expects extra_loc [tokens, >=1].");
        }
        extra_prev_at = extra_at_raw.select(1, 0).reshape({tokens});
    } else if (extra_at_raw.dim() == 1 && extra_at_raw.size(0) == tokens) {
        extra_prev_at = extra_at_raw;
    } else {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_naive expects extra_loc rank 1 or 2.");
    }
    auto positions_at = infinicore::adaptor::to_aten_tensor(positions).reshape({tokens}).to(at::kLong);
    auto write_pos_at = positions_at.remainder(4);

    auto valid_write = write_loc_at.ge(0);
    auto valid_rows = at::nonzero(valid_write).reshape({-1});
    if (valid_rows.numel() > 0) {
        auto valid_groups = write_loc_at.index_select(0, valid_rows);
        auto valid_write_pos = write_pos_at.index_select(0, valid_rows);
        auto valid_values = kv_score_at.index_select(0, valid_rows).reshape({valid_rows.numel(), 4, head_dim});
        state_groups.index_put_({valid_groups, valid_write_pos}, valid_values);
    }

    auto boundary_mask = valid_write.logical_and((positions_at + 1).remainder(4).eq(0));
    auto boundary_rows = at::nonzero(boundary_mask).reshape({-1});
    if (boundary_rows.numel() > 0) {
        auto groups = write_loc_at.index_select(0, boundary_rows);
        auto prev_groups = extra_prev_at.index_select(0, boundary_rows).clamp_min(0);
        auto boundary_positions = positions_at.index_select(0, boundary_rows);

        auto normal_state = state_groups.index_select(0, groups).to(at::kFloat);
        auto overlap_state = state_groups.index_select(0, prev_groups).to(at::kFloat);

        auto overlap_kv = overlap_state.select(2, 0);
        auto normal_kv = normal_state.select(2, 1);
        auto overlap_score = overlap_state.select(2, 2);
        auto normal_score = normal_state.select(2, 3);

        auto has_overlap = boundary_positions.ge(7).view({-1, 1, 1});
        overlap_kv = at::where(has_overlap, overlap_kv, at::zeros_like(overlap_kv));
        overlap_score = at::where(has_overlap, overlap_score, at::full_like(overlap_score, -1.0e9));

        auto ape_at = infinicore::adaptor::to_aten_tensor(ape).contiguous().to(at::kFloat);
        at::Tensor ape_view;
        if (ape_at.dim() == 2 && ape_at.size(0) == 4 && ape_at.size(1) == 2 * head_dim) {
            auto ape_chunks = ape_at.reshape({4, 2, head_dim});
            ape_view = at::cat({ape_chunks.select(1, 1), ape_chunks.select(1, 0)}, 0).contiguous();
        } else if (ape_at.dim() == 2 && ape_at.size(0) == 8 && ape_at.size(1) == head_dim) {
            ape_view = ape_at;
        } else {
            throw std::runtime_error("deepseek_v4_c4_compress_stateful_naive expects ape [4, 2 * head_dim] or [8, head_dim].");
        }

        auto kv_window = at::cat({overlap_kv, normal_kv}, 1);
        auto score_window = at::cat({overlap_score, normal_score}, 1) + ape_view.view({1, 8, head_dim});
        auto prob = at::softmax(score_window, 1);
        auto compressed = (kv_window * prob).sum(1);
        auto output_at = infinicore::adaptor::to_aten_tensor(output);
        output_at.index_copy_(0, boundary_rows, compressed.to(output_at.scalar_type()));
    }
    return output;
#else
    (void)kv_score_input;
    (void)ape;
    (void)compressor_state;
    (void)write_loc;
    (void)extra_loc;
    (void)positions;
    throw std::runtime_error("deepseek_v4_c4_compress_stateful_naive requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}




Tensor deepseek_v4_c128_compress_stateful_naive(const Tensor &kv_score_input,
                                                    const Tensor &ape,
                                                    Tensor compressor_state,
                                                    const Tensor &write_loc,
                                                    const Tensor &positions) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    check_hygon_or_nvidia_tensor(kv_score_input, "deepseek_v4_c128_compress_stateful_naive");
    check_hygon_or_nvidia_tensor(compressor_state, "deepseek_v4_c128_compress_stateful_naive");
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 2 != 0) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_naive expects kv_score_input [tokens, 2 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 2);
    if (head_dim != kDsv4FlashMlaQDim) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_naive expects head_dim 512.");
    }
    if (compressor_state->ndim() != 2 || compressor_state->size(1) != static_cast<size_t>(2 * head_dim) || compressor_state->size(0) % 128 != 0) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_naive expects compressor_state [128 * groups, 2 * head_dim].");
    }
    if (write_loc->numel() != static_cast<size_t>(tokens) || positions->numel() != static_cast<size_t>(tokens)) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_naive metadata token count mismatch.");
    }
    if (ape->ndim() != 2 || ape->size(1) != static_cast<size_t>(head_dim)) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_naive expects ape [128, head_dim].");
    }

    auto output = Tensor::zeros({static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}, kv_score_input->dtype(), kv_score_input->device());
    if (tokens == 0) {
        return output;
    }

    auto state_at = infinicore::adaptor::to_aten_tensor(compressor_state);
    if (!state_at.is_contiguous()) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_naive expects contiguous compressor_state.");
    }
    auto kv_score_at = infinicore::adaptor::to_aten_tensor(kv_score_input).contiguous().to(state_at.scalar_type())
                           .reshape({tokens, 2 * head_dim});
    auto state_groups = state_at.view({static_cast<int64_t>(compressor_state->size(0)) / 128, 128, 2, head_dim});

    auto write_loc_at = infinicore::adaptor::to_aten_tensor(write_loc).reshape({tokens}).to(at::kLong);
    auto positions_at = infinicore::adaptor::to_aten_tensor(positions).reshape({tokens}).to(at::kLong);
    auto write_pos_at = positions_at.remainder(128);

    auto valid_write = write_loc_at.ge(0);
    auto valid_rows = at::nonzero(valid_write).reshape({-1});
    if (valid_rows.numel() > 0) {
        auto valid_groups = write_loc_at.index_select(0, valid_rows);
        auto valid_write_pos = write_pos_at.index_select(0, valid_rows);
        auto valid_values = kv_score_at.index_select(0, valid_rows).reshape({valid_rows.numel(), 2, head_dim});
        state_groups.index_put_({valid_groups, valid_write_pos}, valid_values);
    }

    auto boundary_mask = valid_write.logical_and((positions_at + 1).remainder(128).eq(0));
    auto boundary_rows = at::nonzero(boundary_mask).reshape({-1});
    if (boundary_rows.numel() > 0) {
        auto groups = write_loc_at.index_select(0, boundary_rows);
        auto state = state_groups.index_select(0, groups).to(at::kFloat);
        auto kv_window = state.select(2, 0);
        auto score_window = state.select(2, 1);
        auto ape_at = infinicore::adaptor::to_aten_tensor(ape).contiguous().to(at::kFloat);
        if (ape_at.size(0) < 128) {
            throw std::runtime_error("deepseek_v4_c128_compress_stateful_naive expects ape first dim >= 128.");
        }
        score_window = score_window + ape_at.slice(0, 0, 128).view({1, 128, head_dim});
        auto prob = at::softmax(score_window, 1);
        auto compressed = (kv_window * prob).sum(1);
        auto output_at = infinicore::adaptor::to_aten_tensor(output);
        output_at.index_copy_(0, boundary_rows, compressed.to(output_at.scalar_type()));
    }
    return output;
#else
    (void)kv_score_input;
    (void)ape;
    (void)compressor_state;
    (void)write_loc;
    (void)positions;
    throw std::runtime_error("deepseek_v4_c128_compress_stateful_naive requires an ATen-enabled HYGON build.");
#endif
}



} // namespace infinicore::op
