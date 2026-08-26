#include "infinicore/ops/flash_mla/sparse_decode_fwd.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"

#include "../../../utils.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace {

void check_sparse_decode_common(const Tensor &out,
                                const Tensor &lse,
                                const Tensor &q,
                                const Tensor &k_cache,
                                const Tensor &indices,
                                const char *op_name) {
    if (!out || !lse || !q || !k_cache || !indices) {
        throw std::runtime_error(std::string(op_name) + " expects non-empty out/lse/q/k_cache/indices tensors.");
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
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, lse, q, k_cache, indices);
}

void check_sparse_decode_graph_scheduler_metadata(const std::optional<Tensor> &tile_scheduler_metadata,
                                                  const std::optional<Tensor> &num_splits,
                                                  const Tensor &q,
                                                  const char *op_name) {
    if (!tile_scheduler_metadata.has_value() || !tile_scheduler_metadata.value() || !num_splits.has_value() || !num_splits.value()) {
        throw std::runtime_error(std::string(op_name) + " graph path requires precomputed scheduler metadata.");
    }
    const auto &tile = tile_scheduler_metadata.value();
    const auto &splits = num_splits.value();
    if (tile->ndim() != 2 || tile->size(1) != 8 || splits->ndim() != 1 || splits->size(0) != q->size(0) * q->size(1) + 1) {
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

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SparseDecodeFwd);

common::OpDispatcher<SparseDecodeFwdImplSchema> &sparse_decode_fwd_impl_dispatcher() {
    static common::OpDispatcher<SparseDecodeFwdImplSchema> dispatcher_;
    return dispatcher_;
}

SparseDecodeFwd::SparseDecodeFwd(
    Tensor out,
    Tensor lse,
    const Tensor &q,
    const Tensor &k_cache,
    const Tensor &indices,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> extra_topk_length,
    int64_t head_dim_v,
    double softmax_scale) {
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(),
                                 out,
                                 lse,
                                 q,
                                 k_cache,
                                 indices,
                                 topk_length,
                                 attn_sink,
                                 tile_scheduler_metadata,
                                 num_splits,
                                 extra_k_cache,
                                 extra_indices_in_kvcache,
                                 extra_topk_length,
                                 head_dim_v,
                                 softmax_scale);
}

void SparseDecodeFwd::execute(
    Tensor out,
    Tensor lse,
    const Tensor &q,
    const Tensor &k_cache,
    const Tensor &indices,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> extra_topk_length,
    int64_t head_dim_v,
    double softmax_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SparseDecodeFwd,
                                      out,
                                      lse,
                                      q,
                                      k_cache,
                                      indices,
                                      topk_length,
                                      attn_sink,
                                      tile_scheduler_metadata,
                                      num_splits,
                                      extra_k_cache,
                                      extra_indices_in_kvcache,
                                      extra_topk_length,
                                      head_dim_v,
                                      softmax_scale);
}

// 以下注释不要删除
// 调用过程：
// sparse_decode_fwd          # public API，创建 out/lse
//   -> sparse_decode_fwd_    # out/workspace API，支持 graph
//       -> SparseDecodeFwd::execute
//           -> sparse_decode_fwd_impl    # hygon 目录里的实际实现
//               -> flashmla_sparse_decode_fn(...)
std::tuple<Tensor, Tensor, Tensor, Tensor> sparse_decode_fwd(
    const Tensor &q,
    const Tensor &k_cache,
    const Tensor &indices,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> extra_topk_length,
    int64_t head_dim_v,
    double softmax_scale) {

    if (q->ndim() != 4) {
        throw std::runtime_error("sparse_decode_fwd expects q shape [batch, seq_q, heads, head_dim].");
    }
    if (head_dim_v <= 0) {
        throw std::runtime_error("sparse_decode_fwd expects positive head_dim_v.");
    }

    Tensor out = Tensor::empty({q->size(0), q->size(1), q->size(2), static_cast<size_t>(head_dim_v)},
                               q->dtype(),
                               q->device());
    Tensor lse = Tensor::empty({q->size(0), q->size(2), q->size(1)},
                               DataType::F32,
                               q->device());
    auto [new_tile_scheduler_metadata, new_num_splits] = sparse_decode_fwd_(out,
                                                                            lse,
                                                                            q,
                                                                            k_cache,
                                                                            indices,
                                                                            topk_length,
                                                                            attn_sink,
                                                                            tile_scheduler_metadata,
                                                                            num_splits,
                                                                            extra_k_cache,
                                                                            extra_indices_in_kvcache,
                                                                            extra_topk_length,
                                                                            head_dim_v,
                                                                            softmax_scale);

    return {out, lse, new_tile_scheduler_metadata, new_num_splits};
}

std::tuple<Tensor, Tensor> sparse_decode_fwd_(
    Tensor &out,
    Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    const Tensor &indices,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> extra_topk_length,
    int64_t head_dim_v,
    double softmax_scale) {
    check_sparse_decode_common(out,
                               lse,
                               q,
                               k_cache,
                               indices,
                               "sparse_decode_fwd_");
    if (head_dim_v <= 0 || out->size(3) != static_cast<size_t>(head_dim_v)) {
        throw std::runtime_error("sparse_decode_fwd_ output head_dim_v mismatch.");
    }
    if (context::isGraphRecording()) {
        check_sparse_decode_graph_scheduler_metadata(tile_scheduler_metadata, num_splits, q, "sparse_decode_fwd_");

        SparseDecodeFwd::execute(out,
                                 lse,
                                 q,
                                 k_cache,
                                 indices,
                                 topk_length,
                                 attn_sink,
                                 tile_scheduler_metadata,
                                 num_splits,
                                 extra_k_cache,
                                 extra_indices_in_kvcache,
                                 extra_topk_length,
                                 head_dim_v,
                                 softmax_scale);

        Tensor new_tile_scheduler_metadata = tile_scheduler_metadata.value();
        Tensor new_num_splits = num_splits.value();
        return {new_tile_scheduler_metadata, new_num_splits};
    }

    auto [scheduler_metadata, scheduler_num_splits] = sparse_decode_fwd_impl_dispatcher().lookup(q->device().getType())(out,
                                                                                                                        lse,
                                                                                                                        q,
                                                                                                                        k_cache,
                                                                                                                        indices,
                                                                                                                        topk_length,
                                                                                                                        attn_sink,
                                                                                                                        tile_scheduler_metadata,
                                                                                                                        num_splits,
                                                                                                                        extra_k_cache,
                                                                                                                        extra_indices_in_kvcache,
                                                                                                                        extra_topk_length,
                                                                                                                        head_dim_v,
                                                                                                                        softmax_scale);
    if (!scheduler_metadata || !scheduler_num_splits) {
        throw std::runtime_error("sparse_decode_fwd_ expects non-empty scheduler metadata from sparse_decode_fwd_impl.");
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
