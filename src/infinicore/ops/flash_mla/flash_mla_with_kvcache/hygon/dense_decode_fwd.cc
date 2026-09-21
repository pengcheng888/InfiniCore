#include "infinicore/ops/flash_mla/flash_mla_with_kvcache.hpp"

#include "../../../../adaptor/flashmla/hygon/flashmla_hygon.hpp"
#include "flash_mla_with_kvcache_helper.hpp"
#include "infinicore/context/context.hpp"

#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#endif
#endif

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace infinicore::op {

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

namespace flash_mla::flash_mla_with_kvcache_hygon {

using detail::check_device;
using detail::check_hygon_decode_options;
using detail::check_optional_device;
using detail::copy_flashmla_tensor_exact;
using detail::empty_like;
using detail::has_tensor;
using detail::can_reuse_as_output_buffer;
using detail::resolve_softmax_scale;
using detail::to_aten_tensor_for_flashmla;
using detail::to_optional_aten_for_flashmla;
using detail::to_optional_graph_tensor;
using detail::to_optional_tensor;
using detail::use_sparse_decode;

namespace {

FlashMLASchedMeta make_graph_sched_meta(const FlashMLASchedMeta &metadata) {
    auto graph_metadata = metadata;
    graph_metadata.tile_scheduler_metadata = graph::GraphTensor(metadata.tile_scheduler_metadata);
    graph_metadata.num_splits = graph::GraphTensor(metadata.num_splits);
    return graph_metadata;
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
    std::optional<Tensor> extra_topk_length);

void flash_mla_with_kvcache_impl_internal(
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
    std::optional<Tensor> extra_topk_length,
    std::optional<bool> use_sched_meta_override,
    bool update_sched_meta_state);

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
    bool use_sched_meta;
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

    check_hygon_decode_options(block_table,
                               cache_seqlens,
                               num_splits,
                               is_fp8_kvcache,
                               indices,
                               attn_sink,
                               extra_k_cache,
                               extra_indices_in_kvcache,
                               topk_length,
                               extra_topk_length,
                               "FlashMlaWithKvcache::plan");

    if (!tile_scheduler_metadata.has_sched_buffer()) {
        throw std::runtime_error("FlashMlaWithKvcache::plan requires precomputed scheduler metadata.");
    }

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
                           to_optional_graph_tensor(extra_topk_length),
                           tile_scheduler_metadata.has_valid_sched_meta()};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    flash_mla_with_kvcache_impl_internal(planned->out,
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
                                         planned->use_sched_meta,
                                         false);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::flash_mla::FlashMlaWithKvcache::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    ::infinicore::op::flash_mla::FlashMlaWithKvcache::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    ::infinicore::op::flash_mla::FlashMlaWithKvcache::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    ::infinicore::op::flash_mla::flash_mla_with_kvcache_impl_dispatcher().registerDevice(Device::Type::HYGON, &flash_mla_with_kvcache_impl);
    return true;
}();

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
                                         std::nullopt,
                                         true);
}

void flash_mla_with_kvcache_impl_internal(
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
    std::optional<Tensor> extra_topk_length,
    std::optional<bool> use_sched_meta_override,
    bool update_sched_meta_state) {
    constexpr const char *op_name = "flash_mla_with_kvcache_impl";
    const bool sparse_decode = use_sparse_decode(indices);
    check_hygon_decode_options(block_table,
                               cache_seqlens,
                               num_splits,
                               is_fp8_kvcache,
                               indices,
                               attn_sink,
                               extra_k_cache,
                               extra_indices_in_kvcache,
                               topk_length,
                               extra_topk_length,
                               op_name);
    const bool has_sched_buffer = tile_scheduler_metadata.has_sched_buffer();
    if (static_cast<bool>(tile_scheduler_metadata.tile_scheduler_metadata) != static_cast<bool>(tile_scheduler_metadata.num_splits)) {
        throw std::runtime_error(std::string(op_name) + " expects scheduler metadata and scheduler num_splits to both be set or both be empty.");
    }
    const bool use_sched_meta = use_sched_meta_override.has_value() ? use_sched_meta_override.value() : tile_scheduler_metadata.has_valid_sched_meta();
    const Tensor empty_sched_tensor;
    const Tensor &sched_tile_metadata = use_sched_meta ? tile_scheduler_metadata.tile_scheduler_metadata : empty_sched_tensor;
    const Tensor &sched_num_splits = use_sched_meta ? tile_scheduler_metadata.num_splits : empty_sched_tensor;

    const double scale = resolve_softmax_scale(q, softmax_scale, op_name);

    check_device(out, op_name);
    check_device(lse, op_name);
    check_device(q, op_name);
    check_device(k_cache, op_name);
    if (!sparse_decode) {
        check_device(block_table.value(), op_name);
        check_device(cache_seqlens.value(), op_name);
    }
    if (has_sched_buffer) {
        check_device(tile_scheduler_metadata.tile_scheduler_metadata, op_name);
        check_device(tile_scheduler_metadata.num_splits, op_name);
    }
    check_optional_device(indices, op_name);
    check_optional_device(attn_sink, op_name);
    check_optional_device(extra_k_cache, op_name);
    check_optional_device(extra_indices_in_kvcache, op_name);
    check_optional_device(topk_length, op_name);
    check_optional_device(extra_topk_length, op_name);

#if INFINICORE_TORCH_VERSION_GE_2_11
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#endif

    auto q_flash_at = to_aten_tensor_for_flashmla(q);
    auto k_cache_flash_at = to_aten_tensor_for_flashmla(k_cache);
    std::optional<at::Tensor> tile_scheduler_metadata_flash_at;
    std::optional<at::Tensor> num_splits_flash_at;
    if (sched_tile_metadata) {
        tile_scheduler_metadata_flash_at = to_aten_tensor_for_flashmla(sched_tile_metadata);
        num_splits_flash_at = to_aten_tensor_for_flashmla(sched_num_splits);
    }
    at::Tensor flash_out_at;
    at::Tensor flash_lse_at;
    std::optional<at::Tensor> new_tile_scheduler_metadata;
    std::optional<at::Tensor> new_num_splits;
    if (sparse_decode) {
        auto indices_flash_at = to_aten_tensor_for_flashmla(indices.value());
        auto attn_sink_flash_at = to_optional_aten_for_flashmla(attn_sink);
        auto extra_k_cache_flash_at = to_optional_aten_for_flashmla(extra_k_cache);
        auto extra_indices_flash_at = to_optional_aten_for_flashmla(extra_indices_in_kvcache);
        auto topk_length_flash_at = to_optional_aten_for_flashmla(topk_length);
        auto extra_topk_length_flash_at = to_optional_aten_for_flashmla(extra_topk_length);
        std::tie(flash_out_at, flash_lse_at, new_tile_scheduler_metadata, new_num_splits)
            = infinicore::adaptor::flashmla::hygon::flashmla_sparse_decode_fn(op_name)(
                q_flash_at,
                k_cache_flash_at,
                indices_flash_at,
                topk_length_flash_at,
                attn_sink_flash_at,
                tile_scheduler_metadata_flash_at,
                num_splits_flash_at,
                extra_k_cache_flash_at,
                extra_indices_flash_at,
                extra_topk_length_flash_at,
                static_cast<int>(head_dim_v),
                static_cast<float>(scale));
    } else {
        auto cache_seqlens_flash_at = to_aten_tensor_for_flashmla(cache_seqlens.value());
        auto block_table_flash_at = to_aten_tensor_for_flashmla(block_table.value());
        std::tie(flash_out_at, flash_lse_at, new_tile_scheduler_metadata, new_num_splits)
            = infinicore::adaptor::flashmla::hygon::flashmla_dense_decode_fn(op_name)(
                q_flash_at,
                k_cache_flash_at,
                static_cast<int>(head_dim_v),
                cache_seqlens_flash_at,
                block_table_flash_at,
                static_cast<float>(scale),
                causal,
                tile_scheduler_metadata_flash_at,
                num_splits_flash_at);
    }
    const bool has_new_tile_scheduler_metadata = new_tile_scheduler_metadata.has_value() && new_tile_scheduler_metadata.value().defined();
    const bool has_new_num_splits = new_num_splits.has_value() && new_num_splits.value().defined();

    if (has_new_tile_scheduler_metadata != has_new_num_splits) {
        throw std::runtime_error(std::string(op_name) + " expects vendor returned scheduler metadata and num_splits to both be set or both be empty.");
    }

    if (has_new_tile_scheduler_metadata) {
        auto new_tile_scheduler_metadata_at = new_tile_scheduler_metadata.value().contiguous();
        auto new_num_splits_at = new_num_splits.value().contiguous();
        Tensor new_tile_scheduler_metadata_tensor;
        Tensor new_num_splits_tensor;
        if (can_reuse_as_output_buffer(tile_scheduler_metadata.tile_scheduler_metadata,
                                       new_tile_scheduler_metadata_at,
                                       DataType::I32,
                                       q->device())
            && can_reuse_as_output_buffer(tile_scheduler_metadata.num_splits,
                                          new_num_splits_at,
                                          DataType::I32,
                                          q->device())) {
            new_tile_scheduler_metadata_tensor = tile_scheduler_metadata.tile_scheduler_metadata;
            new_num_splits_tensor = tile_scheduler_metadata.num_splits;
        } else {
            new_tile_scheduler_metadata_tensor = empty_like(new_tile_scheduler_metadata_at, DataType::I32, q->device());
            new_num_splits_tensor = empty_like(new_num_splits_at, DataType::I32, q->device());
        }

        copy_flashmla_tensor_exact(new_tile_scheduler_metadata_tensor, new_tile_scheduler_metadata_at, "tile_scheduler_metadata");
        copy_flashmla_tensor_exact(new_num_splits_tensor, new_num_splits_at, "num_splits");

        if (new_tile_scheduler_metadata_tensor->dtype() != DataType::I32 || new_num_splits_tensor->dtype() != DataType::I32) {
            throw std::runtime_error(std::string(op_name) + " expects vendor returned scheduler metadata tensors to be int32.");
        }

        const size_t expected_num_splits = sparse_decode ? q->size(0) * q->size(1) + 1 : q->size(0) + 1;
        if (new_tile_scheduler_metadata_tensor->ndim() != 2
            || new_tile_scheduler_metadata_tensor->size(1) != 8
            || new_num_splits_tensor->ndim() != 1
            || new_num_splits_tensor->size(0) != expected_num_splits) {
            throw std::runtime_error(std::string(op_name) + " vendor returned scheduler metadata shape mismatch.");
        }

        if (update_sched_meta_state) {
            FlashMLASchedMeta::Config new_config;
            new_config.b = q->size(0);
            new_config.s_q = q->size(1);
            new_config.h_q = q->size(2);
            new_config.page_block_size = k_cache->size(1);
            new_config.h_k = k_cache->size(2);
            new_config.causal = causal;
            new_config.is_fp8_kvcache = is_fp8_kvcache;
            new_config.topk = sparse_decode ? std::make_optional(indices.value()->size(indices.value()->ndim() - 1)) : std::nullopt;
            new_config.extra_page_block_size = has_tensor(extra_k_cache) ? std::make_optional(extra_k_cache.value()->size(1)) : std::nullopt;
            new_config.extra_topk = has_tensor(extra_indices_in_kvcache) ? std::make_optional(extra_indices_in_kvcache.value()->size(extra_indices_in_kvcache.value()->ndim() - 1)) : std::nullopt;

            tile_scheduler_metadata.tile_scheduler_metadata = new_tile_scheduler_metadata_tensor;
            tile_scheduler_metadata.num_splits = new_num_splits_tensor;
            tile_scheduler_metadata.config = new_config;
            tile_scheduler_metadata.have_initialized = true;
            tile_scheduler_metadata.have_refreshed = true;
        }
    }
    copy_flashmla_tensor_exact(out, flash_out_at, "out");
    copy_flashmla_tensor_exact(lse, flash_lse_at, "lse");
}

} // namespace flash_mla::flash_mla_with_kvcache_hygon
#endif

} // namespace infinicore::op
