#include "infinicore/ops/flash_mla/get_mla_decoding_metadata.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include "../../../utils.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace infinicore::op {

namespace {

void check_metadata_inputs(const Tensor &cache_seqlens,
                           int64_t num_q_tokens_per_head_k,
                           int64_t num_heads_k,
                           const char *op_name) {
    if (!cache_seqlens) {
        throw std::runtime_error(std::string(op_name) + " expects non-empty cache_seqlens.");
    }
    if (cache_seqlens->dtype() != DataType::I32 || !cache_seqlens->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous int32 cache_seqlens.");
    }
    if (num_q_tokens_per_head_k <= 0 || num_heads_k <= 0) {
        throw std::runtime_error(std::string(op_name) + " expects positive head counts.");
    }
}

void check_metadata_outputs(const Tensor &tile_scheduler_metadata,
                            const Tensor &num_splits,
                            const Tensor &cache_seqlens,
                            const char *op_name) {
    if (!tile_scheduler_metadata || !num_splits) {
        throw std::runtime_error(std::string(op_name) + " graph mode requires preallocated scheduler metadata outputs.");
    }
    if (tile_scheduler_metadata->ndim() != 2 || tile_scheduler_metadata->size(1) != 8) {
        throw std::runtime_error(std::string(op_name) + " tile_scheduler_metadata shape mismatch.");
    }
    if (num_splits->ndim() != 1 || num_splits->size(0) != cache_seqlens->size(0) + 1) {
        throw std::runtime_error(std::string(op_name) + " num_splits shape mismatch.");
    }
    if (tile_scheduler_metadata->dtype() != DataType::I32 || num_splits->dtype() != DataType::I32) {
        throw std::runtime_error(std::string(op_name) + " scheduler metadata outputs must be int32.");
    }
    if (!tile_scheduler_metadata->is_contiguous() || !num_splits->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " scheduler metadata outputs must be contiguous.");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(tile_scheduler_metadata, num_splits, cache_seqlens);
}

} // namespace

namespace flash_mla {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(GetMlaDecodingMetadata);

common::OpDispatcher<GetMlaDecodingMetadataImplSchema> &get_mla_decoding_metadata_impl_dispatcher() {
    static common::OpDispatcher<GetMlaDecodingMetadataImplSchema> dispatcher_;
    return dispatcher_;
}

GetMlaDecodingMetadata::GetMlaDecodingMetadata(
    Tensor tile_scheduler_metadata,
    Tensor num_splits,
    const Tensor &cache_seqlens,
    int64_t num_q_tokens_per_head_k,
    int64_t num_heads_k,
    std::optional<int64_t> num_heads_q,
    bool is_fp8_kvcache,
    std::optional<int64_t> topk) {
    if (cache_seqlens->device().getType() == Device::Type::METAX) {
        device_graph_capture_safe_ = false;
    }
    INFINICORE_GRAPH_OP_DISPATCH(cache_seqlens->device().getType(),
                                 tile_scheduler_metadata,
                                 num_splits,
                                 cache_seqlens,
                                 num_q_tokens_per_head_k,
                                 num_heads_k,
                                 num_heads_q,
                                 is_fp8_kvcache,
                                 topk);
}

void GetMlaDecodingMetadata::execute(
    Tensor tile_scheduler_metadata,
    Tensor num_splits,
    const Tensor &cache_seqlens,
    int64_t num_q_tokens_per_head_k,
    int64_t num_heads_k,
    std::optional<int64_t> num_heads_q,
    bool is_fp8_kvcache,
    std::optional<int64_t> topk) {
    check_metadata_inputs(cache_seqlens,
                          num_q_tokens_per_head_k,
                          num_heads_k,
                          "GetMlaDecodingMetadata::execute");
    check_metadata_outputs(tile_scheduler_metadata,
                           num_splits,
                           cache_seqlens,
                           "GetMlaDecodingMetadata::execute");

    INFINICORE_GRAPH_OP_RECORD_OR_RUN(GetMlaDecodingMetadata,
                                      tile_scheduler_metadata,
                                      num_splits,
                                      cache_seqlens,
                                      num_q_tokens_per_head_k,
                                      num_heads_k,
                                      num_heads_q,
                                      is_fp8_kvcache,
                                      topk);
}

std::pair<Tensor, Tensor> get_mla_decoding_metadata(
    Tensor tile_scheduler_metadata,
    Tensor num_splits,
    const Tensor &cache_seqlens,
    int64_t num_q_tokens_per_head_k,
    int64_t num_heads_k,
    std::optional<int64_t> num_heads_q,
    bool is_fp8_kvcache,
    std::optional<int64_t> topk) {
    check_metadata_inputs(cache_seqlens,
                          num_q_tokens_per_head_k,
                          num_heads_k,
                          "get_mla_decoding_metadata");

    if (context::isGraphRecording()) {
        if (!tile_scheduler_metadata || !num_splits) {
            throw std::runtime_error("get_mla_decoding_metadata graph mode requires preallocated scheduler metadata outputs.");
        }
        GetMlaDecodingMetadata::execute(tile_scheduler_metadata,
                                        num_splits,
                                        cache_seqlens,
                                        num_q_tokens_per_head_k,
                                        num_heads_k,
                                        num_heads_q,
                                        is_fp8_kvcache,
                                        topk);

        return {tile_scheduler_metadata, num_splits};
    }

    get_mla_decoding_metadata_impl_dispatcher().lookup(cache_seqlens->device().getType())(
        tile_scheduler_metadata,
        num_splits,
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
        num_heads_q,
        is_fp8_kvcache,
        topk);

    return {tile_scheduler_metadata, num_splits};
}

} // namespace flash_mla

} // namespace infinicore::op
