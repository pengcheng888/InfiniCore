#include "infinicore/ops/flash_mla/fwd_kvcache_mla.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
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

void check_fwd_inputs(const Tensor &out,
                      const Tensor &lse,
                      const Tensor &q,
                      const Tensor &k_cache,
                      const std::optional<Tensor> &k_cache_scale,
                      int64_t head_dim_v,
                      const Tensor &cache_seqlens,
                      const Tensor &block_table,
                      const Tensor &tile_scheduler_metadata,
                      const Tensor &num_splits,
                      const std::optional<Tensor> &extra_k_cache,
                      const std::optional<Tensor> &extra_block_table,
                      int64_t cp_world_size,
                      int64_t cp_rank,
                      const std::optional<Tensor> &cp_tot_seqused_k,
                      const char *op_name) {
    if (!out || !lse || !q || !k_cache || !cache_seqlens || !block_table || !tile_scheduler_metadata || !num_splits) {
        throw std::runtime_error(std::string(op_name) + " expects non-empty output/input/metadata tensors.");
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
    if (lse->ndim() != 3 || lse->size(0) != q->size(0) || lse->size(1) != q->size(2) || lse->size(2) != q->size(1)) {
        throw std::runtime_error(std::string(op_name) + " softmax_lse shape mismatch.");
    }
    if (out->dtype() != q->dtype() || lse->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " output dtype mismatch.");
    }
    if (cache_seqlens->dtype() != DataType::I32 || block_table->dtype() != DataType::I32
        || tile_scheduler_metadata->dtype() != DataType::I32 || num_splits->dtype() != DataType::I32) {
        throw std::runtime_error(std::string(op_name) + " metadata tensors must be int32.");
    }
    if (tile_scheduler_metadata->ndim() != 2 || tile_scheduler_metadata->size(1) != 8
        || num_splits->ndim() != 1 || num_splits->size(0) != q->size(0) + 1) {
        throw std::runtime_error(std::string(op_name) + " scheduler metadata shape mismatch.");
    }
    if (!out->is_contiguous() || !lse->is_contiguous() || !q->is_contiguous() || !k_cache->is_contiguous()
        || !cache_seqlens->is_contiguous() || !block_table->is_contiguous()
        || !tile_scheduler_metadata->is_contiguous() || !num_splits->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
    if (cp_world_size <= 0 || cp_rank < 0 || cp_rank >= cp_world_size) {
        throw std::runtime_error(std::string(op_name) + " received invalid cp scalar parameters.");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, lse, q, k_cache, cache_seqlens, block_table, tile_scheduler_metadata, num_splits);
    check_optional_tensor_device(k_cache_scale, q, "k_cache_scale", op_name);
    check_optional_tensor_device(extra_k_cache, q, "extra_k_cache", op_name);
    check_optional_tensor_device(extra_block_table, q, "extra_block_table", op_name);
    check_optional_tensor_device(cp_tot_seqused_k, q, "cp_tot_seqused_k", op_name);
}

} // namespace

namespace flash_mla {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FwdKvcacheMla);

common::OpDispatcher<FwdKvcacheMlaImplSchema> &fwd_kvcache_mla_impl_dispatcher() {
    static common::OpDispatcher<FwdKvcacheMlaImplSchema> dispatcher_;
    return dispatcher_;
}

FwdKvcacheMla::FwdKvcacheMla(
    Tensor out,
    Tensor lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> k_cache_scale,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    const Tensor &tile_scheduler_metadata,
    const Tensor &num_splits,
    bool is_fp8_kvcache,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_block_table,
    int64_t cp_world_size,
    int64_t cp_rank,
    std::optional<Tensor> cp_tot_seqused_k) {
    if (q->device().getType() == Device::Type::METAX) {
        device_graph_capture_safe_ = false;
    }
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(),
                                 out,
                                 lse,
                                 q,
                                 k_cache,
                                 k_cache_scale,
                                 head_dim_v,
                                 cache_seqlens,
                                 block_table,
                                 softmax_scale,
                                 causal,
                                 tile_scheduler_metadata,
                                 num_splits,
                                 is_fp8_kvcache,
                                 extra_k_cache,
                                 extra_block_table,
                                 cp_world_size,
                                 cp_rank,
                                 cp_tot_seqused_k);
}

void FwdKvcacheMla::execute(
    Tensor out,
    Tensor lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> k_cache_scale,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    const Tensor &tile_scheduler_metadata,
    const Tensor &num_splits,
    bool is_fp8_kvcache,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_block_table,
    int64_t cp_world_size,
    int64_t cp_rank,
    std::optional<Tensor> cp_tot_seqused_k) {
    check_fwd_inputs(out,
                     lse,
                     q,
                     k_cache,
                     k_cache_scale,
                     head_dim_v,
                     cache_seqlens,
                     block_table,
                     tile_scheduler_metadata,
                     num_splits,
                     extra_k_cache,
                     extra_block_table,
                     cp_world_size,
                     cp_rank,
                     cp_tot_seqused_k,
                     "FwdKvcacheMla::execute");

    INFINICORE_GRAPH_OP_RECORD_OR_RUN(FwdKvcacheMla,
                                      out,
                                      lse,
                                      q,
                                      k_cache,
                                      k_cache_scale,
                                      head_dim_v,
                                      cache_seqlens,
                                      block_table,
                                      softmax_scale,
                                      causal,
                                      tile_scheduler_metadata,
                                      num_splits,
                                      is_fp8_kvcache,
                                      extra_k_cache,
                                      extra_block_table,
                                      cp_world_size,
                                      cp_rank,
                                      cp_tot_seqused_k);
}

void fwd_kvcache_mla_(
    Tensor &out,
    Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> k_cache_scale,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    const Tensor &tile_scheduler_metadata,
    const Tensor &num_splits,
    bool is_fp8_kvcache,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_block_table,
    int64_t cp_world_size,
    int64_t cp_rank,
    std::optional<Tensor> cp_tot_seqused_k) {
    check_fwd_inputs(out,
                     lse,
                     q,
                     k_cache,
                     k_cache_scale,
                     head_dim_v,
                     cache_seqlens,
                     block_table,
                     tile_scheduler_metadata,
                     num_splits,
                     extra_k_cache,
                     extra_block_table,
                     cp_world_size,
                     cp_rank,
                     cp_tot_seqused_k,
                     "fwd_kvcache_mla_");

    if (context::isGraphRecording()) {
        FwdKvcacheMla::execute(out,
                               lse,
                               q,
                               k_cache,
                               k_cache_scale,
                               head_dim_v,
                               cache_seqlens,
                               block_table,
                               softmax_scale,
                               causal,
                               tile_scheduler_metadata,
                               num_splits,
                               is_fp8_kvcache,
                               extra_k_cache,
                               extra_block_table,
                               cp_world_size,
                               cp_rank,
                               cp_tot_seqused_k);
        return;
    }

    fwd_kvcache_mla_impl_dispatcher().lookup(q->device().getType())(out,
                                                                    lse,
                                                                    q,
                                                                    k_cache,
                                                                    k_cache_scale,
                                                                    head_dim_v,
                                                                    cache_seqlens,
                                                                    block_table,
                                                                    softmax_scale,
                                                                    causal,
                                                                    tile_scheduler_metadata,
                                                                    num_splits,
                                                                    is_fp8_kvcache,
                                                                    extra_k_cache,
                                                                    extra_block_table,
                                                                    cp_world_size,
                                                                    cp_rank,
                                                                    cp_tot_seqused_k);
}

} // namespace flash_mla

} // namespace infinicore::op
