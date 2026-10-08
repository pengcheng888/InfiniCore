#pragma once

#include "infinicore/ops/flash_mla/flash_mla_with_kvcache.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::op::flash_mla::flash_mla_with_kvcache_detail {

inline bool has_tensor(const std::optional<Tensor> &tensor) {
    return tensor.has_value() && tensor.value();
}

inline void check_device(const Tensor &tensor,
                         Device::Type expected_type,
                         const char *expected_name,
                         const char *op_name) {
    if (!tensor || tensor->device().getType() != expected_type) {
        throw std::runtime_error(
            std::string(op_name) + " expects " + expected_name + " tensors.");
    }
}

inline void check_optional_device(const std::optional<Tensor> &tensor,
                                  Device::Type expected_type,
                                  const char *expected_name,
                                  const char *op_name) {
    if (has_tensor(tensor)) {
        check_device(tensor.value(), expected_type, expected_name, op_name);
    }
}

inline double resolve_softmax_scale(const Tensor &q,
                                   const std::optional<double> &softmax_scale,
                                   const char *op_name) {
    if (softmax_scale.has_value()) {
        return softmax_scale.value();
    }
    if (!q || q->ndim() == 0 || q->size(q->ndim() - 1) == 0) {
        throw std::runtime_error(
            std::string(op_name) + " cannot infer softmax_scale from q.");
    }
    return 1.0 / std::sqrt(static_cast<double>(q->size(q->ndim() - 1)));
}

inline void check_optional_tensor_device(const std::optional<Tensor> &tensor,
                                        const Tensor &base,
                                        const char *name,
                                        const char *op_name) {
    if (!has_tensor(tensor)) {
        return;
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(tensor.value(), base);
    if (!tensor.value()->is_contiguous()) {
        throw std::runtime_error(
            std::string(op_name) + " expects contiguous " + name + ".");
    }
}

inline void check_fwd_inputs(
    const Tensor &out,
    const Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    const std::optional<Tensor> &block_table,
    const std::optional<Tensor> &cache_seqlens,
    int64_t head_dim_v,
    FlashMLASchedMeta &tile_scheduler_metadata,
    const std::optional<Tensor> &num_splits,
    bool is_fp8_kvcache,
    const std::optional<Tensor> &indices,
    const std::optional<Tensor> &attn_sink,
    const std::optional<Tensor> &extra_k_cache,
    const std::optional<Tensor> &extra_indices_in_kvcache,
    const std::optional<Tensor> &topk_length,
    const std::optional<Tensor> &extra_topk_length,
    const char *op_name) {
    if (!out || !lse || !q || !k_cache) {
        throw std::runtime_error(
            std::string(op_name)
            + " expects non-empty output and input tensors.");
    }
    const Tensor &sched_tile_metadata =
        tile_scheduler_metadata.tile_scheduler_metadata;
    const Tensor &sched_num_splits = tile_scheduler_metadata.num_splits;
    const bool has_tile_scheduler_metadata =
        static_cast<bool>(sched_tile_metadata);
    const bool has_sched_num_splits = static_cast<bool>(sched_num_splits);
    if (has_tile_scheduler_metadata != has_sched_num_splits) {
        throw std::runtime_error(
            std::string(op_name)
            + " expects scheduler metadata and scheduler num_splits to both be set or both be empty.");
    }
    if (has_tensor(block_table) != has_tensor(cache_seqlens)) {
        throw std::runtime_error(
            std::string(op_name)
            + " expects block_table and cache_seqlens to both be set or both be empty.");
    }
    if (q->ndim() != 4) {
        throw std::runtime_error(
            std::string(op_name)
            + " expects q shape [batch, seq_q, heads, head_dim].");
    }
    if (k_cache->ndim() != 4) {
        throw std::runtime_error(
            std::string(op_name)
            + " expects k_cache shape [blocks, page_size, kv_heads, head_dim].");
    }
    if (out->ndim() != 4 || out->size(0) != q->size(0)
        || out->size(1) != q->size(1) || out->size(2) != q->size(2)) {
        throw std::runtime_error(
            std::string(op_name) + " output shape mismatch.");
    }
    if (head_dim_v <= 0
        || out->size(3) != static_cast<size_t>(head_dim_v)) {
        throw std::runtime_error(
            std::string(op_name) + " output head_dim_v mismatch.");
    }
    if (lse->ndim() != 3 || lse->size(0) != q->size(0)
        || lse->size(1) != q->size(2) || lse->size(2) != q->size(1)) {
        throw std::runtime_error(
            std::string(op_name) + " lse shape mismatch.");
    }
    if (out->dtype() != q->dtype() || lse->dtype() != DataType::F32) {
        throw std::runtime_error(
            std::string(op_name) + " output dtype mismatch.");
    }
    if ((has_tensor(cache_seqlens)
         && cache_seqlens.value()->dtype() != DataType::I32)
        || (has_tensor(block_table)
            && block_table.value()->dtype() != DataType::I32)) {
        throw std::runtime_error(
            std::string(op_name)
            + " cache metadata tensors must be int32.");
    }
    if (has_tile_scheduler_metadata
        && (sched_tile_metadata->dtype() != DataType::I32
            || sched_num_splits->dtype() != DataType::I32)) {
        throw std::runtime_error(
            std::string(op_name)
            + " scheduler metadata tensors must be int32.");
    }
    const bool sparse_fp8_decode = is_fp8_kvcache && has_tensor(indices);
    const size_t expected_num_splits =
        sparse_fp8_decode ? q->size(0) * q->size(1) + 1 : q->size(0) + 1;
    if (tile_scheduler_metadata.has_valid_sched_meta()
        && (sched_tile_metadata->ndim() != 2
            || sched_tile_metadata->size(1) != 8
            || sched_num_splits->ndim() != 1
            || sched_num_splits->size(0) != expected_num_splits)) {
        throw std::runtime_error(
            std::string(op_name) + " scheduler metadata shape mismatch.");
    }
    if ((has_tensor(indices)
         && indices.value()->dtype() != DataType::I32)
        || (has_tensor(topk_length)
            && topk_length.value()->dtype() != DataType::I32)
        || (has_tensor(extra_indices_in_kvcache)
            && extra_indices_in_kvcache.value()->dtype() != DataType::I32)
        || (has_tensor(extra_topk_length)
            && extra_topk_length.value()->dtype() != DataType::I32)) {
        throw std::runtime_error(
            std::string(op_name) + " sparse metadata tensors must be int32.");
    }
    if (has_tensor(attn_sink)
        && attn_sink.value()->dtype() != DataType::F32) {
        throw std::runtime_error(
            std::string(op_name) + " attn_sink must be float32.");
    }
    if (!out->is_contiguous() || !lse->is_contiguous()
        || !q->is_contiguous()
        || (!sparse_fp8_decode && !k_cache->is_contiguous())
        || (has_tile_scheduler_metadata
            && (!sched_tile_metadata->is_contiguous()
                || !sched_num_splits->is_contiguous()))) {
        throw std::runtime_error(
            std::string(op_name) + " expects contiguous tensors.");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, lse, q, k_cache);
    if (has_tile_scheduler_metadata) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
            q, sched_tile_metadata, sched_num_splits);
    }
    check_optional_tensor_device(
        block_table, q, "block_table", op_name);
    check_optional_tensor_device(
        cache_seqlens, q, "cache_seqlens", op_name);
    check_optional_tensor_device(num_splits, q, "num_splits", op_name);
    check_optional_tensor_device(indices, q, "indices", op_name);
    check_optional_tensor_device(attn_sink, q, "attn_sink", op_name);
    check_optional_tensor_device(
        extra_k_cache, q, "extra_k_cache", op_name);
    check_optional_tensor_device(
        extra_indices_in_kvcache, q, "extra_indices_in_kvcache", op_name);
    check_optional_tensor_device(
        topk_length, q, "topk_length", op_name);
    check_optional_tensor_device(
        extra_topk_length, q, "extra_topk_length", op_name);
}

inline std::optional<graph::GraphTensor>
to_optional_graph_tensor(const std::optional<Tensor> &tensor) {
    if (!has_tensor(tensor)) {
        return std::nullopt;
    }
    return graph::GraphTensor(tensor.value());
}

inline std::optional<Tensor>
to_optional_tensor(const std::optional<graph::GraphTensor> &tensor) {
    if (!tensor.has_value()) {
        return std::nullopt;
    }
    return tensor.value();
}

} // namespace infinicore::op::flash_mla::flash_mla_with_kvcache_detail
