#include <atomic>

#include "infinicore/ops/deepseek_v4_sparse_attn_indexer.hpp"

#include "infinicore/device.hpp"
#include "infinicore/ops/deepseek_v4_paged_mqa_logits.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#endif
#endif

#include <dlfcn.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace {

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
constexpr const char *kMqaLogitsSymbol = "_ZN2at6native10mqa_logitsERNS_6TensorES2_S2_S2_S2_iiiiRKSt8optionalIS1_EbS6_";
constexpr const char *kTopKPrefillSymbol = "_ZN2at6native21top_k_per_row_prefillERKNS_6TensorES3_S3_RS1_llll";
constexpr const char *kTopKDecodeSymbol = "_ZN2at6native20top_k_per_row_decodeERKNS_6TensorElS3_RS1_llll";

using MqaLogitsFn = at::Tensor (*)(at::Tensor &,
                                   at::Tensor &,
                                   at::Tensor &,
                                   at::Tensor &,
                                   at::Tensor &,
                                   int,
                                   int,
                                   int,
                                   int,
                                   const std::optional<at::Tensor> &,
                                   bool,
                                   const std::optional<at::Tensor> &);

using TopKPrefillFn = void (*)(const at::Tensor &,
                               const at::Tensor &,
                               const at::Tensor &,
                               at::Tensor &,
                               long,
                               long,
                               long,
                               long);

using TopKDecodeFn = void (*)(const at::Tensor &,
                              long,
                              const at::Tensor &,
                              at::Tensor &,
                              long,
                              long,
                              long,
                              long);

void check_hygon_tensor(const Tensor &tensor, const char *op_name) {
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
}

void *resolve_lightop_symbol(const char *op_name, const char *symbol) {
    void *fn = dlsym(RTLD_DEFAULT, symbol);
    if (fn == nullptr) {
        throw std::runtime_error(
            std::string(op_name) + " requires lightop.op to be loaded with RTLD_GLOBAL; missing symbol: " + symbol);
    }
    return fn;
}

at::Tensor valid_logits_view(const at::Tensor &logits, int64_t rows, int64_t cols) {
    if (logits.dim() != 2) {
        throw std::runtime_error("deepseek_v4_sparse_attn_indexer expects a 2-D logits workspace.");
    }
    if (logits.size(0) < rows || logits.size(1) < cols) {
        throw std::runtime_error("deepseek_v4_sparse_attn_indexer logits workspace is smaller than the logical logits shape.");
    }
    return logits.slice(0, 0, rows).slice(1, 0, cols);
}
#endif

} // namespace

void deepseek_v4_sparse_attn_indexer_prefill_(const Tensor &q,
                                              const Tensor &k,
                                              const Tensor &weights,
                                              const Tensor &cu_seqlen_ks,
                                              const Tensor &cu_seqlen_ke,
                                              Tensor logits,
                                              Tensor topk_indices,
                                              std::optional<Tensor> kv_scale,
                                              int topk_tokens,
                                              bool clean_logits) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    check_hygon_tensor(q, "deepseek_v4_sparse_attn_indexer_prefill_");
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    auto q_at = infinicore::adaptor::to_aten_tensor(q);
    auto k_at = infinicore::adaptor::to_aten_tensor(k);
    auto weights_at = infinicore::adaptor::to_aten_tensor(weights);
    auto cu_seqlen_ks_at = infinicore::adaptor::to_aten_tensor(cu_seqlen_ks);
    auto cu_seqlen_ke_at = infinicore::adaptor::to_aten_tensor(cu_seqlen_ke);
    auto logits_at = infinicore::adaptor::to_aten_tensor(logits);
    auto topk_indices_at = infinicore::adaptor::to_aten_tensor(topk_indices);

    std::optional<at::Tensor> kv_scale_at = std::nullopt;
    at::Tensor kv_scale_storage;
    if (kv_scale.has_value()) {
        kv_scale_storage = infinicore::adaptor::to_aten_tensor(kv_scale.value());
        kv_scale_at = kv_scale_storage;
    }

    if (q_at.dim() != 3 || k_at.dim() != 2) {
        throw std::runtime_error("deepseek_v4_sparse_attn_indexer_prefill_ expects q [M,H,D] and k [N,D].");
    }
    const int q_seq_len = static_cast<int>(q_at.size(0));
    const int kv_seq_len = static_cast<int>(k_at.size(0));
    const int num_heads = static_cast<int>(q_at.size(1));
    const int head_dim = static_cast<int>(q_at.size(2));

    std::optional<at::Tensor> d_out = logits_at;
    static auto mqa_logits_fn = reinterpret_cast<MqaLogitsFn>(
        resolve_lightop_symbol("deepseek_v4_sparse_attn_indexer_prefill_", kMqaLogitsSymbol));
    (void)mqa_logits_fn(q_at,
                        k_at,
                        weights_at,
                        cu_seqlen_ks_at,
                        cu_seqlen_ke_at,
                        q_seq_len,
                        kv_seq_len,
                        num_heads,
                        head_dim,
                        kv_scale_at,
                        clean_logits,
                        d_out);

    auto valid_logits = valid_logits_view(logits_at, q_seq_len, kv_seq_len);
    static auto topk_prefill_fn = reinterpret_cast<TopKPrefillFn>(
        resolve_lightop_symbol("deepseek_v4_sparse_attn_indexer_prefill_", kTopKPrefillSymbol));
    topk_prefill_fn(valid_logits,
                    cu_seqlen_ks_at,
                    cu_seqlen_ke_at,
                    topk_indices_at,
                    valid_logits.size(0),
                    valid_logits.stride(0),
                    valid_logits.stride(1),
                    topk_tokens);
#else
    (void)q;
    (void)k;
    (void)weights;
    (void)cu_seqlen_ks;
    (void)cu_seqlen_ke;
    (void)logits;
    (void)topk_indices;
    (void)kv_scale;
    (void)topk_tokens;
    (void)clean_logits;
    throw std::runtime_error("deepseek_v4_sparse_attn_indexer_prefill_ requires an ATen-enabled HYGON build with lightop.");
#endif
}

void deepseek_v4_sparse_attn_indexer_decode_(const Tensor &q,
                                             const Tensor &fused_kv_cache,
                                             const Tensor &weights,
                                             const Tensor &context_lens,
                                             const Tensor &block_table,
                                             const Tensor &schedule_meta,
                                             Tensor logits,
                                             Tensor topk_indices,
                                             int max_context_len,
                                             int next_n,
                                             int topk_tokens,
                                             bool clean_logits) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    deepseek_v4_paged_mqa_logits_(q,
                                  fused_kv_cache,
                                  weights,
                                  context_lens,
                                  block_table,
                                  schedule_meta,
                                  logits,
                                  max_context_len,
                                  clean_logits);

    check_hygon_tensor(logits, "deepseek_v4_sparse_attn_indexer_decode_");
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    auto logits_at = infinicore::adaptor::to_aten_tensor(logits);
    auto context_lens_at = infinicore::adaptor::to_aten_tensor(context_lens);
    auto topk_indices_at = infinicore::adaptor::to_aten_tensor(topk_indices);

    if (logits_at.dim() != 2) {
        throw std::runtime_error("deepseek_v4_sparse_attn_indexer_decode_ expects a 2-D logits tensor.");
    }

    static auto topk_decode_fn = reinterpret_cast<TopKDecodeFn>(
        resolve_lightop_symbol("deepseek_v4_sparse_attn_indexer_decode_", kTopKDecodeSymbol));
    topk_decode_fn(logits_at,
                   next_n,
                   context_lens_at,
                   topk_indices_at,
                   logits_at.size(0),
                   logits_at.stride(0),
                   logits_at.stride(1),
                   topk_tokens);
#else
    (void)q;
    (void)fused_kv_cache;
    (void)weights;
    (void)context_lens;
    (void)block_table;
    (void)schedule_meta;
    (void)logits;
    (void)topk_indices;
    (void)max_context_len;
    (void)next_n;
    (void)topk_tokens;
    (void)clean_logits;
    throw std::runtime_error("deepseek_v4_sparse_attn_indexer_decode_ requires an ATen-enabled HYGON build with lightop and deepgemm.");
#endif
}

} // namespace infinicore::op
