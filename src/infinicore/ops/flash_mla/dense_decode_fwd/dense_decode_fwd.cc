#include "infinicore/ops/flash_mla/dense_decode_fwd.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"

#include "../../../utils.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace {

void check_dense_decode_common(const Tensor &out,
                               const Tensor &lse,
                               const Tensor &q,
                               const Tensor &k_cache,
                               const Tensor &cache_seqlens,
                               const Tensor &block_table,
                               const char *op_name) {
    if (!out || !lse || !q || !k_cache || !cache_seqlens || !block_table) {
        throw std::runtime_error(std::string(op_name) + " expects non-empty out/lse/q/k_cache/cache_seqlens/block_table tensors.");
    }
    if (q->ndim() != 4) {
        throw std::runtime_error(std::string(op_name) + " expects q shape [batch, seq_q, heads, head_dim].");
    }
    if (out->ndim() != 4 || out->size(0) != q->size(0) || out->size(1) != q->size(1) || out->size(2) != q->size(2)) {
        throw std::runtime_error(std::string(op_name) + " output shape mismatch.");
    }
    if (lse->ndim() != 3 || lse->size(0) != q->size(0) || lse->size(1) != q->size(2) || lse->size(2) != q->size(1)) {
        throw std::runtime_error(std::string(op_name) + " softmax_lse shape mismatch.");
    }
    if (out->dtype() != q->dtype()) {
        throw std::runtime_error(std::string(op_name) + " output dtype must match q dtype.");
    }
    if (lse->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " softmax_lse dtype must be float32.");
    }
    if (!out->is_contiguous() || !lse->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " output tensors must be contiguous.");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, lse, q, k_cache, cache_seqlens, block_table);
}

void check_dense_decode_optional_scheduler_metadata(const std::optional<Tensor> &tile_scheduler_metadata,
                                                    const std::optional<Tensor> &num_splits,
                                                    const Tensor &q,
                                                    const char *op_name) {
    const bool has_tile = tile_scheduler_metadata.has_value() && tile_scheduler_metadata.value();
    const bool has_splits = num_splits.has_value() && num_splits.value();
    if (has_tile != has_splits) {
        throw std::runtime_error(std::string(op_name) + " scheduler metadata must be both set or both empty.");
    }
    if (!has_tile) {
        return;
    }

    const auto &tile = tile_scheduler_metadata.value();
    const auto &splits = num_splits.value();
    if (tile->ndim() != 2 || tile->size(1) != 8 || splits->ndim() != 1 || splits->size(0) != q->size(0) + 1) {
        throw std::runtime_error(std::string(op_name) + " graph scheduler metadata shape mismatch.");
    }
    if (tile->dtype() != DataType::I32 || splits->dtype() != DataType::I32) {
        throw std::runtime_error(std::string(op_name) + " graph scheduler metadata must be int32.");
    }
    if (!tile->is_contiguous() || !splits->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " graph scheduler metadata must be contiguous.");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(tile, splits, q);
}

} // namespace

namespace flash_mla {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DenseDecodeFwd);

common::OpDispatcher<DenseDecodeFwdImplSchema> &dense_decode_fwd_impl_dispatcher() {
    static common::OpDispatcher<DenseDecodeFwdImplSchema> dispatcher_;
    return dispatcher_;
}

DenseDecodeFwd::DenseDecodeFwd(
    Tensor out,
    Tensor lse,
    const Tensor &q,
    const Tensor &k_cache,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits) {
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(),
                                 out,
                                 lse,
                                 q,
                                 k_cache,
                                 head_dim_v,
                                 cache_seqlens,
                                 block_table,
                                 softmax_scale,
                                 causal,
                                 tile_scheduler_metadata,
                                 num_splits);
}

void DenseDecodeFwd::execute(
    Tensor out,
    Tensor lse,
    const Tensor &q,
    const Tensor &k_cache,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits) {
    check_dense_decode_optional_scheduler_metadata(tile_scheduler_metadata, num_splits, q, "DenseDecodeFwd::execute");

    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DenseDecodeFwd,
                                      out,
                                      lse,
                                      q,
                                      k_cache,
                                      head_dim_v,
                                      cache_seqlens,
                                      block_table,
                                      softmax_scale,
                                      causal,
                                      tile_scheduler_metadata,
                                      num_splits);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> dense_decode_fwd(const Tensor &q,
                                                            const Tensor &k_cache,
                                                            int64_t head_dim_v,
                                                            const Tensor &cache_seqlens,
                                                            const Tensor &block_table,
                                                            double softmax_scale,
                                                            bool causal,
                                                            std::optional<Tensor> tile_scheduler_metadata,
                                                            std::optional<Tensor> num_splits) {
    if (q->ndim() != 4) {
        throw std::runtime_error("dense_decode_fwd expects q shape [batch, seq_q, heads, head_dim].");
    }
    if (head_dim_v <= 0) {
        throw std::runtime_error("dense_decode_fwd expects positive head_dim_v.");
    }

    Tensor out = Tensor::empty({q->size(0), q->size(1), q->size(2), static_cast<size_t>(head_dim_v)},
                               q->dtype(),
                               q->device());
    Tensor lse = Tensor::empty({q->size(0), q->size(2), q->size(1)},
                               DataType::F32,
                               q->device());
    auto [new_tile_scheduler_metadata, new_num_splits] = dense_decode_fwd_(out,
                                                                           lse,
                                                                           q,
                                                                           k_cache,
                                                                           head_dim_v,
                                                                           cache_seqlens,
                                                                           block_table,
                                                                           softmax_scale,
                                                                           causal,
                                                                           tile_scheduler_metadata,
                                                                           num_splits);

    return {out, lse, new_tile_scheduler_metadata, new_num_splits};
}

std::tuple<Tensor, Tensor> dense_decode_fwd_(Tensor &out,
                                             Tensor &lse,
                                             const Tensor &q,
                                             const Tensor &k_cache,
                                             int64_t head_dim_v,
                                             const Tensor &cache_seqlens,
                                             const Tensor &block_table,
                                             double softmax_scale,
                                             bool causal,
                                             std::optional<Tensor> tile_scheduler_metadata,
                                             std::optional<Tensor> num_splits) {

    check_dense_decode_common(out, lse, q, k_cache, cache_seqlens, block_table, "dense_decode_fwd_");

    if (head_dim_v <= 0 || out->size(3) != static_cast<size_t>(head_dim_v)) {
        throw std::runtime_error("dense_decode_fwd_ output head_dim_v mismatch.");
    }

    if (context::isGraphRecording()) {
        check_dense_decode_optional_scheduler_metadata(tile_scheduler_metadata, num_splits, q, "dense_decode_fwd_");

        DenseDecodeFwd::execute(out,
                                lse,
                                q,
                                k_cache,
                                head_dim_v,
                                cache_seqlens,
                                block_table,
                                softmax_scale,
                                causal,
                                tile_scheduler_metadata,
                                num_splits);

        // if (tile_scheduler_metadata.has_value() && tile_scheduler_metadata.value()) {
        //     Tensor new_tile_scheduler_metadata = tile_scheduler_metadata.value();
        //     Tensor new_num_splits = num_splits.value();
        //     return {new_tile_scheduler_metadata, new_num_splits};
        // }
        return {Tensor(), Tensor()};
    }

    auto [scheduler_metadata, scheduler_num_splits] = dense_decode_fwd_impl_dispatcher().lookup(q->device().getType())(out,
                                                                                                                       lse,
                                                                                                                       q,
                                                                                                                       k_cache,
                                                                                                                       head_dim_v,
                                                                                                                       cache_seqlens,
                                                                                                                       block_table,
                                                                                                                       softmax_scale,
                                                                                                                       causal,
                                                                                                                       tile_scheduler_metadata,
                                                                                                                       num_splits);
    if (!scheduler_metadata || !scheduler_num_splits) {
        throw std::runtime_error("dense_decode_fwd_ expects non-empty scheduler metadata from dense_decode_fwd_impl.");
    }

    Tensor new_tile_scheduler_metadata = Tensor::empty(scheduler_metadata->shape(),
                                                       scheduler_metadata->dtype(),
                                                       scheduler_metadata->device());
    Tensor new_num_splits = Tensor::empty(scheduler_num_splits->shape(),
                                          scheduler_num_splits->dtype(),
                                          scheduler_num_splits->device());
    new_tile_scheduler_metadata->copy_from(scheduler_metadata);
    new_num_splits->copy_from(scheduler_num_splits);

    return {new_tile_scheduler_metadata, new_num_splits};
}

} // namespace flash_mla

} // namespace infinicore::op
