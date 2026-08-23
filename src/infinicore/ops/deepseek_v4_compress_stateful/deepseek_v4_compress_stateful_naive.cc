#include "infinicore/ops/deepseek_v4_compress_stateful.hpp"

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

constexpr int64_t kDsv4FlashMlaQDim = 512;

} // namespace

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
