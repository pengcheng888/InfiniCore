#include "infinicore/ops/flash_mla/flash_mla_with_kvcache.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"

#include "../../../utils.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace {

void check_optional_tensor_device(const std::optional<Tensor> &tensor,
                                  const Tensor &base,
                                  const char *name,
                                  const char *op_name) {
    if (!tensor.has_value() || !tensor.value()) {
        return;
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(tensor.value(), base);
    if (!tensor.value()->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous " + name + ".");
    }
}

bool has_optional_tensor(const std::optional<Tensor> &tensor) {
    return tensor.has_value() && static_cast<bool>(tensor.value());
}

void check_fwd_inputs(const Tensor &out,
                      const Tensor &lse,
                      const Tensor &q,
                      const Tensor &k_cache,
                      const std::optional<Tensor> &block_table,
                      const std::optional<Tensor> &cache_seqlens,
                      int64_t head_dim_v,
                      const flash_mla::FlashMLASchedMeta &tile_scheduler_metadata,
                      const std::optional<Tensor> &num_splits,
                      const std::optional<Tensor> &indices,
                      const std::optional<Tensor> &attn_sink,
                      const std::optional<Tensor> &extra_k_cache,
                      const std::optional<Tensor> &extra_indices_in_kvcache,
                      const std::optional<Tensor> &topk_length,
                      const std::optional<Tensor> &extra_topk_length,
                      const char *op_name) {
    if (!out || !lse || !q || !k_cache) {
        throw std::runtime_error(std::string(op_name) + " expects non-empty output and input tensors.");
    }
    const Tensor &sched_tile_metadata = tile_scheduler_metadata.tile_scheduler_metadata;
    const Tensor &sched_num_splits = tile_scheduler_metadata.num_splits;
    const bool has_tile_scheduler_metadata = static_cast<bool>(sched_tile_metadata);
    const bool has_sched_num_splits = static_cast<bool>(sched_num_splits);
    if (has_tile_scheduler_metadata != has_sched_num_splits) {
        throw std::runtime_error(std::string(op_name) + " expects scheduler metadata and scheduler num_splits to both be set or both be empty.");
    }
    if (has_optional_tensor(block_table) != has_optional_tensor(cache_seqlens)) {
        throw std::runtime_error(std::string(op_name) + " expects block_table and cache_seqlens to both be set or both be empty.");
    }
    if (q->ndim() != 4) {
        throw std::runtime_error(std::string(op_name) + " expects q shape [batch, seq_q, heads, head_dim].");
    }
    if (k_cache->ndim() != 4) {
        throw std::runtime_error(std::string(op_name) + " expects k_cache shape [blocks, page_size, kv_heads, head_dim].");
    }
    if (out->ndim() != 4 || out->size(0) != q->size(0) || out->size(1) != q->size(1) || out->size(2) != q->size(2)) {
        throw std::runtime_error(std::string(op_name) + " output shape mismatch.");
    }
    if (head_dim_v <= 0 || out->size(3) != static_cast<size_t>(head_dim_v)) {
        throw std::runtime_error(std::string(op_name) + " output head_dim_v mismatch.");
    }
    if (lse->ndim() != 3 || lse->size(0) != q->size(0)
        || lse->size(1) != q->size(2) || lse->size(2) != q->size(1)) {
        throw std::runtime_error(std::string(op_name) + " lse shape mismatch.");
    }
    if (out->dtype() != q->dtype() || lse->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " output dtype mismatch.");
    }
    if ((has_optional_tensor(cache_seqlens) && cache_seqlens.value()->dtype() != DataType::I32)
        || (has_optional_tensor(block_table) && block_table.value()->dtype() != DataType::I32)) {
        throw std::runtime_error(std::string(op_name) + " cache metadata tensors must be int32.");
    }
    if (has_tile_scheduler_metadata
        && (sched_tile_metadata->dtype() != DataType::I32 || sched_num_splits->dtype() != DataType::I32)) {
        throw std::runtime_error(std::string(op_name) + " scheduler metadata tensors must be int32.");
    }
    if (has_tile_scheduler_metadata
        && (sched_tile_metadata->ndim() != 2 || sched_tile_metadata->size(1) != 8
            || sched_num_splits->ndim() != 1 || sched_num_splits->size(0) != q->size(0) + 1)) {
        throw std::runtime_error(std::string(op_name) + " scheduler metadata shape mismatch.");
    }
    if (!out->is_contiguous() || !lse->is_contiguous() || !q->is_contiguous() || !k_cache->is_contiguous()
        || (has_tile_scheduler_metadata && (!sched_tile_metadata->is_contiguous() || !sched_num_splits->is_contiguous()))) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, lse, q, k_cache);
    if (has_tile_scheduler_metadata) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, sched_tile_metadata, sched_num_splits);
    }
    check_optional_tensor_device(block_table, q, "block_table", op_name);
    check_optional_tensor_device(cache_seqlens, q, "cache_seqlens", op_name);
    check_optional_tensor_device(num_splits, q, "num_splits", op_name);
    check_optional_tensor_device(indices, q, "indices", op_name);
    check_optional_tensor_device(attn_sink, q, "attn_sink", op_name);
    check_optional_tensor_device(extra_k_cache, q, "extra_k_cache", op_name);
    check_optional_tensor_device(extra_indices_in_kvcache, q, "extra_indices_in_kvcache", op_name);
    check_optional_tensor_device(topk_length, q, "topk_length", op_name);
    check_optional_tensor_device(extra_topk_length, q, "extra_topk_length", op_name);
}

} // namespace

namespace flash_mla {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FlashMlaWithKvcache);

common::OpDispatcher<FlashMlaWithKvcacheImplSchema> &flash_mla_with_kvcache_impl_dispatcher() {
    static common::OpDispatcher<FlashMlaWithKvcacheImplSchema> dispatcher_;
    return dispatcher_;
}

FlashMlaWithKvcache::FlashMlaWithKvcache(
    Tensor out,
    Tensor lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> block_table,
    std::optional<Tensor> cache_seqlens,
    int64_t head_dim_v,
    const FlashMLASchedMeta &tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    std::optional<double> softmax_scale,
    bool causal,
    bool is_fp8_kvcache,
    std::optional<Tensor> indices,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> extra_topk_length) {
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(),
                                 out,
                                 lse,
                                 q,
                                 k_cache,
                                 block_table,
                                 cache_seqlens,
                                 head_dim_v,
                                 tile_scheduler_metadata,
                                 num_splits,
                                 softmax_scale,
                                 causal,
                                 is_fp8_kvcache,
                                 indices,
                                 attn_sink,
                                 extra_k_cache,
                                 extra_indices_in_kvcache,
                                 topk_length,
                                 extra_topk_length);
}

void FlashMlaWithKvcache::execute(
    Tensor out,
    Tensor lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> block_table,
    std::optional<Tensor> cache_seqlens,
    int64_t head_dim_v,
    const FlashMLASchedMeta &tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    std::optional<double> softmax_scale,
    bool causal,
    bool is_fp8_kvcache,
    std::optional<Tensor> indices,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> extra_topk_length) {
    check_fwd_inputs(out,
                     lse,
                     q,
                     k_cache,
                     block_table,
                     cache_seqlens,
                     head_dim_v,
                     tile_scheduler_metadata,
                     num_splits,
                     indices,
                     attn_sink,
                     extra_k_cache,
                     extra_indices_in_kvcache,
                     topk_length,
                     extra_topk_length,
                     "FlashMlaWithKvcache::execute");

    if (!tile_scheduler_metadata.has_sched_buffer()) {
        throw std::runtime_error("FlashMlaWithKvcache graph execution requires precomputed scheduler metadata.");
    }

    INFINICORE_GRAPH_OP_RECORD_OR_RUN(FlashMlaWithKvcache,
                                      out,
                                      lse,
                                      q,
                                      k_cache,
                                      block_table,
                                      cache_seqlens,
                                      head_dim_v,
                                      tile_scheduler_metadata,
                                      num_splits,
                                      softmax_scale,
                                      causal,
                                      is_fp8_kvcache,
                                      indices,
                                      attn_sink,
                                      extra_k_cache,
                                      extra_indices_in_kvcache,
                                      topk_length,
                                      extra_topk_length);
}

std::pair<Tensor, Tensor> flash_mla_with_kvcache(
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> block_table,
    std::optional<Tensor> cache_seqlens,
    int64_t head_dim_v,
    const FlashMLASchedMeta &tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    std::optional<double> softmax_scale,
    bool causal,
    bool is_fp8_kvcache,
    std::optional<Tensor> indices,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> extra_topk_length) {
    constexpr const char *op_name = "flash_mla_with_kvcache";
    if (!q) {
        throw std::runtime_error(std::string(op_name) + " expects a non-empty q tensor.");
    }
    if (q->ndim() != 4) {
        throw std::runtime_error(std::string(op_name) + " expects q shape [batch, seq_q, heads, head_dim].");
    }
    if (head_dim_v <= 0) {
        throw std::runtime_error(std::string(op_name) + " expects positive head_dim_v.");
    }

    auto out = Tensor::empty(
        {q->size(0), q->size(1), q->size(2), static_cast<size_t>(head_dim_v)},
        q->dtype(),
        q->device());
    auto lse = Tensor::empty(
        {q->size(0), q->size(2), q->size(1)},
        DataType::F32,
        q->device());

    check_fwd_inputs(out,
                     lse,
                     q,
                     k_cache,
                     block_table,
                     cache_seqlens,
                     head_dim_v,
                     tile_scheduler_metadata,
                     num_splits,
                     indices,
                     attn_sink,
                     extra_k_cache,
                     extra_indices_in_kvcache,
                     topk_length,
                     extra_topk_length,
                     op_name);

    if (context::isGraphRecording()) {
        FlashMlaWithKvcache::execute(out,
                                     lse,
                                     q,
                                     k_cache,
                                     block_table,
                                     cache_seqlens,
                                     head_dim_v,
                                     tile_scheduler_metadata,
                                     num_splits,
                                     softmax_scale,
                                     causal,
                                     is_fp8_kvcache,
                                     indices,
                                     attn_sink,
                                     extra_k_cache,
                                     extra_indices_in_kvcache,
                                     topk_length,
                                     extra_topk_length);
    } else {
        flash_mla_with_kvcache_impl_dispatcher().lookup(q->device().getType())(out,
                                                                               lse,
                                                                               q,
                                                                               k_cache,
                                                                               block_table,
                                                                               cache_seqlens,
                                                                               head_dim_v,
                                                                               tile_scheduler_metadata,
                                                                               num_splits,
                                                                               softmax_scale,
                                                                               causal,
                                                                               is_fp8_kvcache,
                                                                               indices,
                                                                               attn_sink,
                                                                               extra_k_cache,
                                                                               extra_indices_in_kvcache,
                                                                               topk_length,
                                                                               extra_topk_length);
    }
    return {out, lse};
}

} // namespace flash_mla

} // namespace infinicore::op
