#if defined(ENABLE_ILUVATAR_VENDOR_OPS)

#include "infinicore/ops/flash_mla/flash_mla_with_kvcache.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include "../flash_mla_with_kvcache_helper.hpp"

#include <ATen/ATen.h>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::op::flash_mla::flash_mla_with_kvcache_iluvatar {
namespace {

namespace flash_mla_checks = ::infinicore::op::flash_mla::flash_mla_with_kvcache_detail;
using flash_mla_checks::has_tensor;

void check_device(const Tensor &tensor, const char *op_name) {
    flash_mla_checks::check_device(tensor, Device::Type::ILUVATAR, "ILUVATAR", op_name);
}

void check_optional_device(const std::optional<Tensor> &tensor,
                           const char *op_name) {
    flash_mla_checks::check_optional_device(tensor, Device::Type::ILUVATAR, "ILUVATAR", op_name);
}

using flash_mla_checks::resolve_softmax_scale;

void check_iluvatar_options(const std::optional<Tensor> &block_table,
                            const std::optional<Tensor> &cache_seqlens,
                            const std::optional<Tensor> &num_splits,
                            bool causal,
                            bool is_fp8_kvcache,
                            const std::optional<Tensor> &indices,
                            const std::optional<Tensor> &attn_sink,
                            const std::optional<Tensor> &extra_k_cache,
                            const std::optional<Tensor> &extra_indices_in_kvcache,
                            const std::optional<Tensor> &topk_length,
                            const std::optional<Tensor> &extra_topk_length,
                            const char *op_name) {
    if (has_tensor(num_splits)) {
        throw std::runtime_error(std::string(op_name) + " does not support num_splits override on ILUVATAR.");
    }
    if (causal) {
        throw std::runtime_error(std::string(op_name) + " requires causal=false on ILUVATAR MLA decode.");
    }
    if (is_fp8_kvcache) {
        throw std::runtime_error(std::string(op_name) + " does not support is_fp8_kvcache=true on ILUVATAR.");
    }

    if (has_tensor(indices)) {
        if (has_tensor(block_table) || has_tensor(cache_seqlens)) {
            throw std::runtime_error(std::string(op_name) + " sparse decode does not use block_table/cache_seqlens on ILUVATAR.");
        }
        if (!has_tensor(topk_length)) {
            throw std::runtime_error(std::string(op_name) + " sparse decode requires topk_length on ILUVATAR.");
        }
    } else {
        if (!has_tensor(block_table) || !has_tensor(cache_seqlens)) {
            throw std::runtime_error(std::string(op_name) + " requires block_table and cache_seqlens on ILUVATAR.");
        }
        if (has_tensor(attn_sink) || has_tensor(topk_length)) {
            throw std::runtime_error(std::string(op_name) + " attn_sink/topk_length require sparse indices on ILUVATAR.");
        }
    }

    if (has_tensor(extra_k_cache) || has_tensor(extra_indices_in_kvcache)
        || has_tensor(extra_topk_length)) {
        throw std::runtime_error(std::string(op_name) + " does not support extended sparse KV caches on ILUVATAR.");
    }
}

void validate_inputs(const Tensor &out,
                     const Tensor &lse,
                     const Tensor &q,
                     const Tensor &k_cache,
                     const Tensor &block_table,
                     const Tensor &cache_seqlens,
                     int64_t head_dim_v,
                     const char *op_name) {
    check_device(out, op_name);
    check_device(lse, op_name);
    check_device(q, op_name);
    check_device(k_cache, op_name);
    check_device(block_table, op_name);
    check_device(cache_seqlens, op_name);

    if (q->ndim() != 4 || q->size(1) != 1) {
        throw std::runtime_error(std::string(op_name) + " expects decode q shape [batch, 1, heads, head_dim].");
    }
    if (k_cache->ndim() != 4 || k_cache->size(2) != 1) {
        throw std::runtime_error(std::string(op_name) + " expects MLA KV cache [blocks, page_size, 1, head_dim].");
    }
    if (out->ndim() != 4 || out->size(0) != q->size(0) || out->size(1) != 1
        || out->size(2) != q->size(2) || out->size(3) != static_cast<size_t>(head_dim_v)) {
        throw std::runtime_error(std::string(op_name) + " output shape mismatch.");
    }
    if (head_dim_v <= 0 || static_cast<size_t>(head_dim_v) > k_cache->size(3)) {
        throw std::runtime_error(std::string(op_name) + " invalid head_dim_v.");
    }
    if (lse->ndim() != 3 || lse->size(0) != q->size(0) || lse->size(1) != q->size(2)
        || lse->size(2) != 1 || lse->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " lse shape or dtype mismatch.");
    }
    if (q->dtype() != out->dtype() || k_cache->dtype() != q->dtype()
        || (q->dtype() != DataType::F16 && q->dtype() != DataType::BF16)) {
        throw std::runtime_error(std::string(op_name) + " requires matching fp16/bfloat16 q, cache, and output.");
    }
    if (block_table->dtype() != DataType::I32 || cache_seqlens->dtype() != DataType::I32
        || block_table->ndim() != 2 || block_table->size(0) != q->size(0)
        || cache_seqlens->ndim() != 1 || cache_seqlens->size(0) != q->size(0)) {
        throw std::runtime_error(std::string(op_name) + " expects int32 block_table [batch, blocks] and cache_seqlens [batch].");
    }
    if (!out->is_contiguous() || !lse->is_contiguous() || !q->is_contiguous()
        || !k_cache->is_contiguous() || !block_table->is_contiguous()
        || !cache_seqlens->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
}

void validate_sparse_inputs(const Tensor &out,
                            const Tensor &lse,
                            const Tensor &q,
                            const Tensor &k_cache,
                            const Tensor &indices,
                            const std::optional<Tensor> &attn_sink,
                            const Tensor &topk_length,
                            int64_t head_dim_v,
                            const char *op_name) {
    check_device(out, op_name);
    check_device(lse, op_name);
    check_device(q, op_name);
    check_device(k_cache, op_name);
    check_device(indices, op_name);
    check_device(topk_length, op_name);
    check_optional_device(attn_sink, op_name);

    if (q->ndim() != 4 || q->size(1) != 1) {
        throw std::runtime_error(std::string(op_name) + " expects sparse decode q [tokens, 1, heads, head_dim].");
    }
    if (k_cache->ndim() != 4 || k_cache->size(2) != 1
        || k_cache->size(3) != q->size(3)) {
        throw std::runtime_error(std::string(op_name) + " expects sparse KV cache [blocks, page_size, 1, head_dim].");
    }
    if (out->ndim() != 4 || out->size(0) != q->size(0) || out->size(1) != 1
        || out->size(2) != q->size(2) || out->size(3) != static_cast<size_t>(head_dim_v)) {
        throw std::runtime_error(std::string(op_name) + " sparse output shape mismatch.");
    }
    if (head_dim_v != 512) {
        throw std::runtime_error(std::string(op_name) + " sparse decode requires head_dim_v=512 on ILUVATAR.");
    }
    if (q->size(3) != 512 && q->size(3) != 576 && q->size(3) != 704) {
        throw std::runtime_error(std::string(op_name) + " sparse decode requires head_dim 512/576/704 on ILUVATAR.");
    }
    if (lse->ndim() != 3 || lse->size(0) != q->size(0) || lse->size(1) != q->size(2)
        || lse->size(2) != 1 || lse->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " sparse lse shape or dtype mismatch.");
    }
    if (q->dtype() != out->dtype() || k_cache->dtype() != q->dtype()
        || (q->dtype() != DataType::F16 && q->dtype() != DataType::BF16)) {
        throw std::runtime_error(std::string(op_name) + " requires matching fp16/bfloat16 sparse q, cache, and output.");
    }
    if (indices->ndim() != 3 || indices->dtype() != DataType::I32
        || indices->size(0) != q->size(0) || indices->size(1) != 1) {
        throw std::runtime_error(std::string(op_name) + " expects int32 indices [tokens, 1, topk].");
    }
    if (topk_length->ndim() != 1 || topk_length->dtype() != DataType::I32
        || topk_length->size(0) != q->size(0)) {
        throw std::runtime_error(std::string(op_name) + " expects int32 topk_length [tokens].");
    }
    if (has_tensor(attn_sink)
        && (attn_sink.value()->ndim() != 1 || attn_sink.value()->dtype() != DataType::F32
            || attn_sink.value()->size(0) != q->size(2))) {
        throw std::runtime_error(std::string(op_name) + " expects float32 attn_sink [heads].");
    }
    if (!out->is_contiguous() || !lse->is_contiguous() || !q->is_contiguous()
        || !k_cache->is_contiguous() || !indices->is_contiguous()
        || !topk_length->is_contiguous()
        || (has_tensor(attn_sink) && !attn_sink.value()->is_contiguous())) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous sparse tensors.");
    }
}

using flash_mla_checks::to_optional_graph_tensor;
using flash_mla_checks::to_optional_tensor;

FlashMLASchedMeta make_graph_sched_meta(const FlashMLASchedMeta &metadata) {
    auto graph_metadata = metadata;
    if (metadata.tile_scheduler_metadata) {
        graph_metadata.tile_scheduler_metadata = graph::GraphTensor(metadata.tile_scheduler_metadata);
    }
    if (metadata.num_splits) {
        graph_metadata.num_splits = graph::GraphTensor(metadata.num_splits);
    }
    return graph_metadata;
}

void flash_mla_with_kvcache_impl_internal(
    Tensor &out,
    Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> block_table,
    std::optional<Tensor> cache_seqlens,
    int64_t head_dim_v,
    FlashMLASchedMeta &,
    std::optional<Tensor> num_splits,
    std::optional<double> softmax_scale,
    bool causal,
    bool is_fp8_kvcache,
    std::optional<Tensor> indices,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> extra_topk_length,
    bool use_cuda_graph) {
    constexpr const char *op_name = "flash_mla_with_kvcache_impl";
    check_iluvatar_options(block_table,
                           cache_seqlens,
                           num_splits,
                           causal,
                           is_fp8_kvcache,
                           indices,
                           attn_sink,
                           extra_k_cache,
                           extra_indices_in_kvcache,
                           topk_length,
                           extra_topk_length,
                           op_name);

    if (has_tensor(indices)) {
        if (!has_tensor(topk_length)) {
            // check_iluvatar_options already enforces this; keep the branch
            // below independent of optional unwrapping.
            throw std::runtime_error(std::string(op_name) + " sparse decode requires topk_length.");
        }
        validate_sparse_inputs(out,
                               lse,
                               q,
                               k_cache,
                               indices.value(),
                               attn_sink,
                               topk_length.value(),
                               head_dim_v,
                               op_name);
        if (!adaptor::iluvatar_vendor::sparse_flash_mla_available()) {
            throw std::runtime_error(std::string(op_name) + " requires the ILUVATAR flash_mla_sparse_v2 vendor extension.");
        }

        auto out_3d = out->squeeze(1);
        auto q_3d = q->squeeze(1);
        auto kv_3d = k_cache->view(
            {k_cache->size(0) * k_cache->size(1), 1, k_cache->size(3)});
        std::optional<at::Tensor> sink_at;
        if (has_tensor(attn_sink)) {
            sink_at = adaptor::to_aten_tensor(attn_sink.value());
        }

        // The v2 sparse kernel does not expose LSE. The common API still
        // requires an F32 tensor; InfiniLM's DSV4 path ignores it, so keep the
        // contract explicit and deterministic by filling it with zeros.
        adaptor::set_aten_stream_to_infinicore();
        adaptor::to_aten_tensor(lse).zero_();

        auto out_at = adaptor::to_aten_tensor(out_3d);
        auto q_at = adaptor::to_aten_tensor(q_3d);
        auto kv_at = adaptor::to_aten_tensor(kv_3d);
        auto indices_at = adaptor::to_aten_tensor(indices.value());
        auto topk_length_at = adaptor::to_aten_tensor(topk_length.value());
        adaptor::iluvatar_vendor::sparse_flash_mla(
            out_at,
            q_at,
            kv_at,
            indices_at,
            topk_length_at,
            static_cast<float>(resolve_softmax_scale(q, softmax_scale, op_name)),
            sink_at);
        return;
    }

    const Tensor &block_table_tensor = block_table.value();
    const Tensor &cache_seqlens_tensor = cache_seqlens.value();
    validate_inputs(out,
                    lse,
                    q,
                    k_cache,
                    block_table_tensor,
                    cache_seqlens_tensor,
                    head_dim_v,
                    op_name);
    if (!adaptor::iluvatar_vendor::paged_attention_mla_available()) {
        throw std::runtime_error(std::string(op_name) + " requires the ILUVATAR paged_attention_mla vendor extension.");
    }

    auto out_3d = out->squeeze(1);
    auto q_3d = q->squeeze(1);
    auto k_cache_3d = k_cache->squeeze(2);
    auto lse_2d = lse->squeeze(2);
    const int64_t max_context_len = static_cast<int64_t>(block_table_tensor->size(1) * k_cache->size(1));

    auto out_at = adaptor::to_aten_tensor(out_3d);
    auto q_at = adaptor::to_aten_tensor(q_3d);
    auto k_cache_at = adaptor::to_aten_tensor(k_cache_3d);
    auto block_table_at = adaptor::to_aten_tensor(block_table_tensor);
    auto cache_seqlens_at = adaptor::to_aten_tensor(cache_seqlens_tensor);
    auto lse_at = adaptor::to_aten_tensor(lse_2d);

    if (k_cache->size(1) == 64) {
        throw std::runtime_error(std::string(op_name)
                                 + " does not support ILUVATAR block_size=64; use --block-size 16 or another vendor-supported page size.");
    }

    adaptor::iluvatar_vendor::paged_attention_mla(
        out_at,
        q_at,
        k_cache_at,
        resolve_softmax_scale(q, softmax_scale, op_name),
        block_table_at,
        cache_seqlens_at,
        max_context_len,
        use_cuda_graph,
        lse_at);
}

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor lse;
    graph::GraphTensor q;
    graph::GraphTensor k_cache;
    std::optional<graph::GraphTensor> block_table;
    std::optional<graph::GraphTensor> cache_seqlens;
    int64_t head_dim_v;
    FlashMLASchedMeta tile_scheduler_metadata;
    std::optional<graph::GraphTensor> num_splits;
    std::optional<double> softmax_scale;
    bool causal;
    bool is_fp8_kvcache;
    std::optional<graph::GraphTensor> indices;
    std::optional<graph::GraphTensor> attn_sink;
    std::optional<graph::GraphTensor> extra_k_cache;
    std::optional<graph::GraphTensor> extra_indices_in_kvcache;
    std::optional<graph::GraphTensor> topk_length;
    std::optional<graph::GraphTensor> extra_topk_length;
};

void *plan(Tensor out,
           Tensor lse,
           const Tensor &q,
           const Tensor &k_cache,
           std::optional<Tensor> block_table,
           std::optional<Tensor> cache_seqlens,
           int64_t head_dim_v,
           FlashMLASchedMeta &tile_scheduler_metadata,
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
    check_iluvatar_options(block_table,
                           cache_seqlens,
                           num_splits,
                           causal,
                           is_fp8_kvcache,
                           indices,
                           attn_sink,
                           extra_k_cache,
                           extra_indices_in_kvcache,
                           topk_length,
                           extra_topk_length,
                           "FlashMlaWithKvcache::plan");

    return new PlannedMeta{graph::GraphTensor(out),
                           graph::GraphTensor(lse),
                           graph::GraphTensor(q),
                           graph::GraphTensor(k_cache),
                           to_optional_graph_tensor(block_table),
                           to_optional_graph_tensor(cache_seqlens),
                           head_dim_v,
                           make_graph_sched_meta(tile_scheduler_metadata),
                           to_optional_graph_tensor(num_splits),
                           softmax_scale,
                           causal,
                           is_fp8_kvcache,
                           to_optional_graph_tensor(indices),
                           to_optional_graph_tensor(attn_sink),
                           to_optional_graph_tensor(extra_k_cache),
                           to_optional_graph_tensor(extra_indices_in_kvcache),
                           to_optional_graph_tensor(topk_length),
                           to_optional_graph_tensor(extra_topk_length)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    flash_mla_with_kvcache_impl_internal(
        planned->out,
        planned->lse,
        planned->q,
        planned->k_cache,
        to_optional_tensor(planned->block_table),
        to_optional_tensor(planned->cache_seqlens),
        planned->head_dim_v,
        planned->tile_scheduler_metadata,
        to_optional_tensor(planned->num_splits),
        planned->softmax_scale,
        planned->causal,
        planned->is_fp8_kvcache,
        to_optional_tensor(planned->indices),
        to_optional_tensor(planned->attn_sink),
        to_optional_tensor(planned->extra_k_cache),
        to_optional_tensor(planned->extra_indices_in_kvcache),
        to_optional_tensor(planned->topk_length),
        to_optional_tensor(planned->extra_topk_length),
        true);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace

void flash_mla_with_kvcache_impl(
    Tensor &out,
    Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> block_table,
    std::optional<Tensor> cache_seqlens,
    int64_t head_dim_v,
    FlashMLASchedMeta &tile_scheduler_metadata,
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
    flash_mla_with_kvcache_impl_internal(out,
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
                                         extra_topk_length,
                                         false);
}

const bool registered = []() {
    FlashMlaWithKvcache::plan_dispatcher().registerDevice(Device::Type::ILUVATAR, &plan);
    FlashMlaWithKvcache::run_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    FlashMlaWithKvcache::cleanup_dispatcher().registerDevice(Device::Type::ILUVATAR, &cleanup);
    flash_mla_with_kvcache_impl_dispatcher().registerDevice(Device::Type::ILUVATAR, &flash_mla_with_kvcache_impl);
    return true;
}();

} // namespace infinicore::op::flash_mla::flash_mla_with_kvcache_iluvatar

#endif // ENABLE_ILUVATAR_VENDOR_OPS
