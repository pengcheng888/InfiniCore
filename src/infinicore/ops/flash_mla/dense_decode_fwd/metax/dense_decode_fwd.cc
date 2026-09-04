#include "infinicore/ops/flash_mla/dense_decode_fwd.hpp"

#include "../../fwd_kvcache_mla/metax/fwd_kvcache_mla.hpp"
#include "../../get_mla_decoding_metadata/metax/get_mla_decoding_metadata.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#endif

#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::op {

#if defined(ENABLE_ATEN) && defined(ENABLE_METAX_API)
namespace {

void check_device(const Tensor &tensor, const char *op_name) {
    if (!tensor || tensor->device().getType() != Device::Type::METAX) {
        throw std::runtime_error(std::string(op_name) + " expects METAX tensors in this build.");
    }
}

void check_optional_device(const std::optional<Tensor> &tensor, const char *op_name) {
    if (tensor.has_value() && tensor.value()) {
        check_device(*tensor, op_name);
    }
}

void copy_tensor_exact(Tensor &dst, const Tensor &src, const char *name) {
    if (!dst || !src) {
        throw std::runtime_error(std::string("dense_decode_fwd_impl: ") + name + " expects non-empty tensors.");
    }
    if (dst->shape() != src->shape()) {
        throw std::runtime_error(std::string("dense_decode_fwd_impl: ") + name + " shape mismatch.");
    }
    if (dst->dtype() != src->dtype()) {
        throw std::runtime_error(std::string("dense_decode_fwd_impl: ") + name + " dtype mismatch.");
    }
    if (dst->device() != src->device()) {
        throw std::runtime_error(std::string("dense_decode_fwd_impl: ") + name + " device mismatch.");
    }
    if (!dst->is_contiguous() || !src->is_contiguous()) {
        throw std::runtime_error(std::string("dense_decode_fwd_impl: ") + name + " must be contiguous.");
    }
    auto dst_at = infinicore::adaptor::to_aten_tensor(dst);
    auto src_at = infinicore::adaptor::to_aten_tensor(src);
    dst_at.copy_(src_at);
}

std::optional<graph::GraphTensor> to_optional_graph_tensor(const std::optional<Tensor> &tensor) {
    if (!tensor.has_value() || !tensor.value()) {
        return std::nullopt;
    }
    return graph::GraphTensor(tensor.value());
}

std::optional<Tensor> to_optional_tensor(const std::optional<graph::GraphTensor> &tensor) {
    if (!tensor.has_value()) {
        return std::nullopt;
    }
    return tensor.value();
}

} // namespace

namespace flash_mla::dense_decode_fwd_metax {

void dense_decode_fwd_impl(
    Tensor &out,
    Tensor &lse,
    std::optional<Tensor> &new_tile_scheduler_metadata,
    std::optional<Tensor> &new_num_splits,
    const Tensor &q,
    const Tensor &k_cache,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits);

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor lse;
    std::optional<graph::GraphTensor> new_tile_scheduler_metadata;
    std::optional<graph::GraphTensor> new_num_splits;
    graph::GraphTensor q;
    graph::GraphTensor k_cache;
    int64_t head_dim_v;
    graph::GraphTensor cache_seqlens;
    graph::GraphTensor block_table;
    double softmax_scale;
    bool causal;
    std::optional<graph::GraphTensor> tile_scheduler_metadata;
    std::optional<graph::GraphTensor> num_splits;
};

void *plan(Tensor out,
           Tensor lse,
           std::optional<Tensor> new_tile_scheduler_metadata,
           std::optional<Tensor> new_num_splits,
           const Tensor &q,
           const Tensor &k_cache,
           int64_t head_dim_v,
           const Tensor &cache_seqlens,
           const Tensor &block_table,
           double softmax_scale,
           bool causal,
           std::optional<Tensor> tile_scheduler_metadata,
           std::optional<Tensor> num_splits) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(lse),
        to_optional_graph_tensor(new_tile_scheduler_metadata),
        to_optional_graph_tensor(new_num_splits),
        graph::GraphTensor(q),
        graph::GraphTensor(k_cache),
        head_dim_v,
        graph::GraphTensor(cache_seqlens),
        graph::GraphTensor(block_table),
        softmax_scale,
        causal,
        to_optional_graph_tensor(tile_scheduler_metadata),
        to_optional_graph_tensor(num_splits)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    std::optional<Tensor> new_tile_scheduler_metadata = to_optional_tensor(planned->new_tile_scheduler_metadata);
    std::optional<Tensor> new_num_splits = to_optional_tensor(planned->new_num_splits);
    std::optional<Tensor> tile_scheduler_metadata = to_optional_tensor(planned->tile_scheduler_metadata);
    std::optional<Tensor> num_splits = to_optional_tensor(planned->num_splits);

    dense_decode_fwd_impl(planned->out,
                          planned->lse,
                          new_tile_scheduler_metadata,
                          new_num_splits,
                          planned->q,
                          planned->k_cache,
                          planned->head_dim_v,
                          planned->cache_seqlens,
                          planned->block_table,
                          planned->softmax_scale,
                          planned->causal,
                          tile_scheduler_metadata,
                          num_splits);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::flash_mla::DenseDecodeFwd::plan_dispatcher().registerDevice(Device::Type::METAX, &plan);
    ::infinicore::op::flash_mla::DenseDecodeFwd::run_dispatcher().registerDevice(Device::Type::METAX, &run);
    ::infinicore::op::flash_mla::DenseDecodeFwd::cleanup_dispatcher().registerDevice(Device::Type::METAX, &cleanup);
    ::infinicore::op::flash_mla::dense_decode_fwd_impl_dispatcher().registerDevice(Device::Type::METAX, &dense_decode_fwd_impl);
    return true;
}();

void dense_decode_fwd_impl(
    Tensor &out,
    Tensor &lse,
    std::optional<Tensor> &new_tile_scheduler_metadata_out,
    std::optional<Tensor> &new_num_splits_out,
    const Tensor &q,
    const Tensor &k_cache,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits) {
    constexpr const char *op_name = "dense_decode_fwd_impl";
    check_device(q, op_name);
    check_device(k_cache, op_name);
    check_device(cache_seqlens, op_name);
    check_device(block_table, op_name);
    check_optional_device(new_tile_scheduler_metadata_out, op_name);
    check_optional_device(new_num_splits_out, op_name);
    check_optional_device(tile_scheduler_metadata, op_name);
    check_optional_device(num_splits, op_name);

    if (k_cache->ndim() != 4) {
        throw std::runtime_error("dense_decode_fwd_impl: METAX FlashMLA expects k_cache shape [blocks, page_size, kv_heads, head_dim].");
    }
    if (cache_seqlens->dtype() != DataType::I32 || block_table->dtype() != DataType::I32) {
        throw std::runtime_error("dense_decode_fwd_impl: METAX FlashMLA expects int32 cache_seqlens and block_table.");
    }
    if (!q->is_contiguous() || !k_cache->is_contiguous() || !cache_seqlens->is_contiguous() || !block_table->is_contiguous()) {
        throw std::runtime_error("dense_decode_fwd_impl: METAX FlashMLA expects contiguous q/k_cache/cache_seqlens/block_table.");
    }
    const int64_t kv_heads = static_cast<int64_t>(k_cache->size(2));
    if (kv_heads <= 0) {
        throw std::runtime_error("dense_decode_fwd_impl: k_cache kv_heads must be positive.");
    }
    const int64_t total_q_heads = static_cast<int64_t>(q->size(1) * q->size(2));
    if (total_q_heads % kv_heads != 0) {
        throw std::runtime_error("dense_decode_fwd_impl: q heads must be divisible by kv_heads.");
    }
    const int64_t num_q_tokens_per_head_k = total_q_heads / kv_heads;

    const bool has_metadata = tile_scheduler_metadata.has_value() && tile_scheduler_metadata.value()
                           && num_splits.has_value() && num_splits.value();
    if ((tile_scheduler_metadata.has_value() && tile_scheduler_metadata.value())
        != (num_splits.has_value() && num_splits.value())) {
        throw std::runtime_error("dense_decode_fwd_impl: scheduler metadata inputs must be both set or both empty.");
    }

    if (!has_metadata) {
        Tensor new_tile_scheduler_metadata = new_tile_scheduler_metadata_out.has_value()
                                               ? new_tile_scheduler_metadata_out.value()
                                               : Tensor{};
        Tensor new_num_splits = new_num_splits_out.has_value()
                                  ? new_num_splits_out.value()
                                  : Tensor{};
        get_mla_decoding_metadata_metax::get_mla_decoding_metadata_impl(
            new_tile_scheduler_metadata,
            new_num_splits,
            cache_seqlens,
            num_q_tokens_per_head_k,
            kv_heads,
            std::nullopt,
            false,
            std::nullopt);
        new_tile_scheduler_metadata_out = new_tile_scheduler_metadata;
        new_num_splits_out = new_num_splits;
        tile_scheduler_metadata = new_tile_scheduler_metadata_out;
        num_splits = new_num_splits_out;
    } else {
        if (!new_tile_scheduler_metadata_out.has_value() || !new_tile_scheduler_metadata_out.value()) {
            new_tile_scheduler_metadata_out = tile_scheduler_metadata.value();
        } else {
            copy_tensor_exact(new_tile_scheduler_metadata_out.value(),
                              tile_scheduler_metadata.value(),
                              "new_tile_scheduler_metadata");
        }
        if (!new_num_splits_out.has_value() || !new_num_splits_out.value()) {
            new_num_splits_out = num_splits.value();
        } else {
            copy_tensor_exact(new_num_splits_out.value(),
                              num_splits.value(),
                              "new_num_splits");
        }
    }

    fwd_kvcache_mla_metax::fwd_kvcache_mla_impl(out,
                                                lse,
                                                q,
                                                k_cache,
                                                std::nullopt,
                                                head_dim_v,
                                                cache_seqlens,
                                                block_table,
                                                softmax_scale,
                                                causal,
                                                tile_scheduler_metadata.value(),
                                                num_splits.value(),
                                                false,
                                                std::nullopt,
                                                std::nullopt,
                                                1,
                                                0,
                                                std::nullopt);
}

} // namespace flash_mla::dense_decode_fwd_metax
#endif

} // namespace infinicore::op
