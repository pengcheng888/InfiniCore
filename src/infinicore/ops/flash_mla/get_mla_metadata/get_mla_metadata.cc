#include "infinicore/ops/flash_mla/get_mla_metadata.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"

#include "../../../utils.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op::flash_mla {

namespace {

void check_inputs(const Tensor &cache_seqlens,
                  const FlashMLASchedMeta &sched_meta,
                  int64_t num_q_tokens_per_head_k,
                  int64_t num_heads_k,
                  std::optional<int64_t> num_heads_q,
                  std::optional<int64_t> topk,
                  const char *op_name) {
    if (!cache_seqlens) {
        throw std::runtime_error(std::string(op_name) + " expects non-empty cache_seqlens.");
    }
    if (cache_seqlens->dtype() != DataType::I32
        || cache_seqlens->ndim() != 1
        || !cache_seqlens->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous int32 cache_seqlens [batch].");
    }
    if (num_q_tokens_per_head_k <= 0 || num_heads_k <= 0
        || (num_heads_q.has_value() && num_heads_q.value() <= 0)
        || (topk.has_value() && topk.value() <= 0)) {
        throw std::runtime_error(std::string(op_name) + " expects positive metadata parameters.");
    }

    const bool has_tile_scheduler_metadata = static_cast<bool>(sched_meta.tile_scheduler_metadata);
    const bool has_num_splits = static_cast<bool>(sched_meta.num_splits);
    if (has_tile_scheduler_metadata != has_num_splits) {
        throw std::runtime_error(std::string(op_name) + " expects tile_scheduler_metadata and num_splits to both be set or both be empty.");
    }
    if (!has_tile_scheduler_metadata) {
        return;
    }
    if (sched_meta.tile_scheduler_metadata->dtype() != DataType::I32
        || sched_meta.tile_scheduler_metadata->ndim() != 2
        || sched_meta.tile_scheduler_metadata->size(1) != 8
        || sched_meta.num_splits->dtype() != DataType::I32
        || sched_meta.num_splits->ndim() != 1
        || sched_meta.num_splits->size(0) != cache_seqlens->size(0) + 1
        || !sched_meta.tile_scheduler_metadata->is_contiguous()
        || !sched_meta.num_splits->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " scheduler metadata buffer mismatch.");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        cache_seqlens,
        sched_meta.tile_scheduler_metadata,
        sched_meta.num_splits);
}

} // namespace

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(GetMlaMetadata);

common::OpDispatcher<GetMlaMetadataImplSchema> &get_mla_metadata_impl_dispatcher() {
    static common::OpDispatcher<GetMlaMetadataImplSchema> dispatcher;
    return dispatcher;
}

GetMlaMetadata::GetMlaMetadata(const Tensor &cache_seqlens,
                               FlashMLASchedMeta &sched_meta,
                               int64_t num_q_tokens_per_head_k,
                               int64_t num_heads_k,
                               std::optional<int64_t> num_heads_q,
                               bool is_fp8_kvcache,
                               std::optional<int64_t> topk) {
    INFINICORE_GRAPH_OP_DISPATCH(cache_seqlens->device().getType(),
                                 cache_seqlens,
                                 sched_meta,
                                 num_q_tokens_per_head_k,
                                 num_heads_k,
                                 num_heads_q,
                                 is_fp8_kvcache,
                                 topk);
}

void GetMlaMetadata::execute(const Tensor &cache_seqlens,
                             FlashMLASchedMeta &sched_meta,
                             int64_t num_q_tokens_per_head_k,
                             int64_t num_heads_k,
                             std::optional<int64_t> num_heads_q,
                             bool is_fp8_kvcache,
                             std::optional<int64_t> topk) {
    check_inputs(cache_seqlens,
                 sched_meta,
                 num_q_tokens_per_head_k,
                 num_heads_k,
                 num_heads_q,
                 topk,
                 "GetMlaMetadata::execute");
    if (!sched_meta.has_sched_buffer()) {
        throw std::runtime_error(
            "GetMlaMetadata::execute requires preallocated scheduler metadata buffers.");
    }

    INFINICORE_GRAPH_OP_RECORD_OR_RUN(GetMlaMetadata,
                                      cache_seqlens,
                                      sched_meta,
                                      num_q_tokens_per_head_k,
                                      num_heads_k,
                                      num_heads_q,
                                      is_fp8_kvcache,
                                      topk);
}

void get_mla_metadata_(
    FlashMLASchedMeta &sched_meta,
    const Tensor &cache_seqlens,
    int64_t num_q_tokens_per_head_k,
    int64_t num_heads_k,
    std::optional<int64_t> num_heads_q,
    bool is_fp8_kvcache,
    std::optional<int64_t> topk) {
    check_inputs(cache_seqlens,
                 sched_meta,
                 num_q_tokens_per_head_k,
                 num_heads_k,
                 num_heads_q,
                 topk,
                 "get_mla_metadata_");

    if (context::isGraphRecording()) {
        GetMlaMetadata::execute(cache_seqlens,
                                sched_meta,
                                num_q_tokens_per_head_k,
                                num_heads_k,
                                num_heads_q,
                                is_fp8_kvcache,
                                topk);
    } else {
        get_mla_metadata_impl_dispatcher()
            .lookup(cache_seqlens->device().getType())(cache_seqlens,
                                                       sched_meta,
                                                       num_q_tokens_per_head_k,
                                                       num_heads_k,
                                                       num_heads_q,
                                                       is_fp8_kvcache,
                                                       topk);
    }
}

} // namespace infinicore::op::flash_mla
