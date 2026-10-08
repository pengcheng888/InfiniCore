#include "infinicore/ops/flash_mla/flash_mla_with_kvcache.hpp"
#include "flash_mla_with_kvcache_helper.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"

#include "../../../utils.hpp"

#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace {
namespace flash_mla_checks = flash_mla::flash_mla_with_kvcache_detail;
using flash_mla_checks::check_fwd_inputs;


} // namespace

namespace flash_mla {

// Keep the graph dispatchers before the eager impl dispatcher. Reordering
// these definitions reproduced a double free during MetaX TP8 teardown.
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FlashMlaWithKvcache);

common::OpDispatcher<FlashMlaWithKvcacheImplSchema> &flash_mla_with_kvcache_impl_dispatcher() {
    static common::OpDispatcher<FlashMlaWithKvcacheImplSchema> dispatcher_;
    return dispatcher_;
}

FlashMlaWithKvcache::FlashMlaWithKvcache(Tensor out,
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

void FlashMlaWithKvcache::execute(Tensor out,
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

    // execute is a public graph entry point and must validate its own inputs.
    check_fwd_inputs(out,
                     lse,
                     q,
                     k_cache,
                     block_table,
                     cache_seqlens,
                     head_dim_v,
                     tile_scheduler_metadata,
                     num_splits,
                     is_fp8_kvcache,
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

    // TODO(hygon): Graph recording only plans the op; it does not run the
    // vendor kernel, so we cannot prove that the scheduler metadata buffer has
    // been refreshed for the current step here. Keep later layers on the
    // no-reuse path for now. Once Hygon exposes a metadata refresh path that
    // writes the preallocated buffer during graph replay, mark the shared
    // sched meta valid after the producer op so subsequent layers can reuse it.
}

std::pair<Tensor, Tensor> flash_mla_with_kvcache(
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

    auto out = Tensor::empty({q->size(0), q->size(1), q->size(2), static_cast<size_t>(head_dim_v)}, q->dtype(), q->device());
    auto lse = Tensor::empty({q->size(0), q->size(2), q->size(1)}, DataType::F32, q->device());

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
        check_fwd_inputs(out,
                         lse,
                         q,
                         k_cache,
                         block_table,
                         cache_seqlens,
                         head_dim_v,
                         tile_scheduler_metadata,
                         num_splits,
                         is_fp8_kvcache,
                         indices,
                         attn_sink,
                         extra_k_cache,
                         extra_indices_in_kvcache,
                         topk_length,
                         extra_topk_length,
                         op_name);
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
