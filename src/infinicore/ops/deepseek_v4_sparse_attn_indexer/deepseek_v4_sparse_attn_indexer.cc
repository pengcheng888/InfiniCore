#include <atomic>
#include <cmath>
#include <limits>

#include "infinicore/ops/deepseek_v4_sparse_attn_indexer.hpp"

#include "deepseek_v4_sparse_attn_indexer_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/ops/deepseek_v4_paged_mqa_logits.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#endif
#endif

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4C4SparseAttnIndexerNoLogits);

namespace {

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
constexpr const char *kMqaLogitsSymbol = "_ZN2at6native10mqa_logitsERNS_6TensorES2_S2_S2_S2_iiiiRKSt8optionalIS1_EbS6_";
constexpr const char *kTopKPrefillSymbol = "_ZN2at6native21top_k_per_row_prefillERKNS_6TensorES3_S3_RS1_llll";
constexpr const char *kTopKDecodeSymbol = "_ZN2at6native20top_k_per_row_decodeERKNS_6TensorElS3_RS1_llll";
constexpr const char *kLightopPagedMqaLogitsSymbol = "_ZN7lightop9deep_gemm16paged_mqa_logitsERKN2at6TensorES4_S4_S4_S4_RSt8optionalIS2_ERKiRKb";

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

using PagedMqaLogitsFn = at::Tensor (*)(const at::Tensor &,
                                        const at::Tensor &,
                                        const at::Tensor &,
                                        const at::Tensor &,
                                        const at::Tensor &,
                                        std::optional<at::Tensor> &,
                                        const int &,
                                        const bool &);

void check_hygon_tensor(const Tensor &tensor, const char *op_name) {
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
}

template <typename Fn>
Fn checked_symbol(void *handle, const char *symbol) {
    dlerror();
    void *fn = dlsym(handle, symbol);
    const char *error = dlerror();
    if (error != nullptr || fn == nullptr) {
        throw std::runtime_error(std::string("lightop SO is missing required symbol ") + symbol +
                                 (error != nullptr ? std::string(": ") + error : ""));
    }
    return reinterpret_cast<Fn>(fn);
}

void *open_lightop_so() {
    std::vector<std::string> candidates;
    if (const char *env_path = std::getenv("INFINICORE_LIGHTOP_OP_SO")) {
        if (env_path[0] != '\0') {
            candidates.emplace_back(env_path);
        }
    }
    candidates.emplace_back("/usr/local/lib/python3.10/dist-packages/lightop/op.cpython-310-x86_64-linux-gnu.so");
    candidates.emplace_back("/usr/local/lib/python3.11/dist-packages/lightop/op.cpython-311-x86_64-linux-gnu.so");
    candidates.emplace_back("op.cpython-310-x86_64-linux-gnu.so");
    candidates.emplace_back("op.cpython-311-x86_64-linux-gnu.so");

    std::ostringstream errors;
    for (const auto &path : candidates) {
        void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle != nullptr) {
            return handle;
        }
        if (const char *error = dlerror()) {
            errors << "\n  " << path << ": " << error;
        }
    }
    throw std::runtime_error("failed to load lightop op SO. Set INFINICORE_LIGHTOP_OP_SO to lightop/op*.so." + errors.str());
}

struct LightopSymbols {
    void *handle{nullptr};
    MqaLogitsFn mqa_logits{nullptr};
    TopKPrefillFn topk_prefill{nullptr};
    TopKDecodeFn topk_decode{nullptr};
    PagedMqaLogitsFn paged_mqa_logits{nullptr};
};

const LightopSymbols &lightop_symbols() {
    static LightopSymbols symbols;
    static std::once_flag once;
    std::call_once(once, [] {
        symbols.handle = open_lightop_so();
        symbols.mqa_logits = checked_symbol<MqaLogitsFn>(symbols.handle, kMqaLogitsSymbol);
        symbols.topk_prefill = checked_symbol<TopKPrefillFn>(symbols.handle, kTopKPrefillSymbol);
        symbols.topk_decode = checked_symbol<TopKDecodeFn>(symbols.handle, kTopKDecodeSymbol);
        symbols.paged_mqa_logits = checked_symbol<PagedMqaLogitsFn>(symbols.handle, kLightopPagedMqaLogitsSymbol);
    });
    return symbols;
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


constexpr int64_t kC4IndexerHeadDim = 128;
constexpr int64_t kC4IndexerScaleBytes = 4;
constexpr int64_t kC4TopK = 512;
constexpr double kC4IndexerFp8Max = 448.0;

at::ScalarType c4_indexer_fp8_dtype() {
    return at::ScalarType::Float8_e4m3fn;
}

at::Tensor squeeze_last_if_needed(const at::Tensor &tensor) {
    if (tensor.dim() == 2 && tensor.size(1) == 1) {
        return tensor.squeeze(1);
    }
    return tensor;
}

at::Tensor as_c4_seq_lens_for_gemm(const at::Tensor &seq_lens) {
    auto flat = squeeze_last_if_needed(seq_lens);
    if (flat.dim() != 1) {
        throw std::runtime_error("deepseek_v4_c4_sparse_attn_indexer_ expects c4_seq_lens [batch] or [batch, 1].");
    }
    return flat.to(at::kInt).unsqueeze(1);
}

at::Tensor as_c4_seq_lens_flat(const at::Tensor &seq_lens) {
    auto flat = squeeze_last_if_needed(seq_lens);
    if (flat.dim() != 1) {
        throw std::runtime_error("deepseek_v4_c4_sparse_attn_indexer_ expects c4_seq_lens [batch] or [batch, 1].");
    }
    return flat.to(at::kInt);
}

std::pair<at::Tensor, at::Tensor> c4_act_quant(const at::Tensor &q) {
    if (q.dim() != 3 || q.size(2) != kC4IndexerHeadDim) {
        throw std::runtime_error("deepseek_v4_c4_sparse_attn_indexer_ expects q [batch, heads, 128].");
    }
    auto q_contig = q.contiguous();
    const auto sizes = q_contig.sizes();
    auto q_float = q_contig.reshape({-1, kC4IndexerHeadDim}).to(at::kFloat);
    auto scale = at::clamp_min(at::amax(at::abs(q_float), {-1}, true), 1.0e-4) / kC4IndexerFp8Max;
    auto q_fp8 = at::clamp(q_float / scale, -kC4IndexerFp8Max, kC4IndexerFp8Max)
                     .to(c4_indexer_fp8_dtype())
                     .reshape(sizes);
    auto q_scale = scale.reshape({sizes[0], sizes[1], 1});
    return {q_fp8, q_scale};
}

int c4_scalar_type_for_kernel(const Tensor &tensor, const char *op_name) {
    if (tensor->dtype() == DataType::BF16) {
        return deepseek_v4_sparse_attn_indexer::kDsv4BF16;
    }
    if (tensor->dtype() == DataType::F16) {
        return deepseek_v4_sparse_attn_indexer::kDsv4F16;
    }
    if (tensor->dtype() == DataType::F32) {
        return deepseek_v4_sparse_attn_indexer::kDsv4F32;
    }
    throw std::runtime_error(std::string(op_name) + " supports bf16/fp16/fp32 tensors.");
}

int c4_at_scalar_type_for_kernel(const at::Tensor &tensor, const char *op_name) {
    if (tensor.scalar_type() == at::kBFloat16) {
        return deepseek_v4_sparse_attn_indexer::kDsv4BF16;
    }
    if (tensor.scalar_type() == at::kHalf) {
        return deepseek_v4_sparse_attn_indexer::kDsv4F16;
    }
    if (tensor.scalar_type() == at::kFloat) {
        return deepseek_v4_sparse_attn_indexer::kDsv4F32;
    }
    throw std::runtime_error(std::string(op_name) + " supports bf16/fp16/fp32 tensors.");
}

void *current_accelerator_stream() {
#if defined(ENABLE_HYGON_API)
    return reinterpret_cast<void *>(infinicore::adaptor::get_hip_stream().stream());
#else
    return nullptr;
#endif
}

at::Tensor c4_fused_weights(const at::Tensor &weights, const at::Tensor &q_scale, float weight_scale) {
    if (weights.dim() != 2 || q_scale.dim() != 3 || q_scale.size(2) != 1 ||
        weights.size(0) != q_scale.size(0) || weights.size(1) != q_scale.size(1)) {
        throw std::runtime_error("deepseek_v4_c4_sparse_attn_indexer_ expects weights [batch, heads] matching q scale.");
    }
    return weights.contiguous().to(at::kFloat) * static_cast<double>(weight_scale) * q_scale.squeeze(2).to(at::kFloat);
}

void topk_transform_512_at(const at::Tensor &scores,
                           const at::Tensor &seq_lens,
                           const at::Tensor &page_tables,
                           at::Tensor &out_page_indices,
                           int page_size) {
    if (scores.dim() != 2 || page_tables.dim() != 2 || out_page_indices.dim() != 2) {
        throw std::runtime_error("topk_transform_512 expects scores/page_tables/out_page_indices to be 2-D.");
    }
    const int64_t batch = scores.size(0);
    const int64_t max_seq_len = scores.size(1);
    if (seq_lens.dim() != 1 || seq_lens.size(0) != batch || page_tables.size(0) != batch || out_page_indices.size(0) != batch || out_page_indices.size(1) < kC4TopK) {
        throw std::runtime_error("topk_transform_512 shape mismatch.");
    }
    if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
        throw std::runtime_error("topk_transform_512 page_size must be a positive power of two.");
    }

    auto int_opts = out_page_indices.options().dtype(at::kInt);
    auto long_opts = page_tables.options().dtype(at::kLong);
    auto seq = seq_lens.to(at::kInt);
    auto sequential = at::arange(kC4TopK, int_opts).unsqueeze(0).expand({batch, kC4TopK});
    auto negative = at::full({batch, kC4TopK}, -1, int_opts);
    auto sequential_valid = sequential < seq.unsqueeze(1);

    at::Tensor raw_indices;
    at::Tensor valid_topk;
    if (max_seq_len <= kC4TopK) {
        raw_indices = at::where(sequential_valid, sequential, negative);
        valid_topk = sequential_valid;
    } else {
        auto positions = at::arange(max_seq_len, scores.options().dtype(at::kLong)).unsqueeze(0);
        auto valid_mask = positions < seq.to(at::kLong).unsqueeze(1);
        auto masked_scores = scores.masked_fill(~valid_mask, -std::numeric_limits<float>::infinity());
        auto topk_result = at::topk(masked_scores, kC4TopK, 1, true, false);
        raw_indices = std::get<1>(topk_result).to(at::kInt);
        auto gathered_scores = scores.gather(1, raw_indices.to(at::kLong));
        valid_topk = gathered_scores.ne(-std::numeric_limits<float>::infinity());
        auto needs_sequential = (seq <= kC4TopK).unsqueeze(1);
        raw_indices = at::where(needs_sequential, at::where(sequential_valid, sequential, negative), raw_indices);
        valid_topk = at::where(needs_sequential, sequential_valid, valid_topk);
    }

    auto raw_long = raw_indices.to(at::kLong);
    auto page_idx = at::floor_divide(raw_long, page_size);
    auto offset_in_page = at::remainder(raw_long, page_size);
    auto page_idx_clamped = at::clamp_min(page_idx, 0);
    auto physical_pages = page_tables.to(at::kLong).gather(1, page_idx_clamped);
    auto page_indices = (physical_pages * page_size + offset_in_page).to(at::kInt);
    auto transformed = at::where(valid_topk, page_indices, negative);
    out_page_indices.slice(1, 0, kC4TopK).copy_(transformed);
}

void topk_transform_512_dispatch_at(const at::Tensor &scores,
                                    const at::Tensor &seq_lens,
                                    const at::Tensor &page_tables,
                                    at::Tensor &out_page_indices,
                                    int page_size,
                                    const char *op_name) {
    if (scores.size(1) > kC4TopK) {
        topk_transform_512_at(scores, seq_lens, page_tables, out_page_indices, page_size);
        return;
    }
    if (!scores.is_contiguous() || !seq_lens.is_contiguous() || !page_tables.is_contiguous() || !out_page_indices.is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " native topk path expects contiguous tensors.");
    }
    deepseek_v4_sparse_attn_indexer::launch_topk_transform_512(
        reinterpret_cast<const float *>(scores.data_ptr()),
        scores.stride(0),
        seq_lens.data_ptr(),
        seq_lens.scalar_type() == at::kLong,
        page_tables.data_ptr(),
        page_tables.scalar_type() == at::kLong,
        page_tables.stride(0),
        reinterpret_cast<int32_t *>(out_page_indices.data_ptr()),
        out_page_indices.stride(0),
        scores.size(0),
        scores.size(1),
        page_size,
        current_accelerator_stream());
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
    auto mqa_logits_fn = lightop_symbols().mqa_logits;
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
    auto topk_prefill_fn = lightop_symbols().topk_prefill;
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

    auto topk_decode_fn = lightop_symbols().topk_decode;
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



void deepseek_v4_c4_sparse_attn_indexer_(const Tensor &q,
                                         const Tensor &indexer_weights,
                                         const Tensor &indexer_kv_cache_raw,
                                         const Tensor &c4_seq_lens,
                                         const Tensor &page_table,
                                         Tensor logits,
                                         Tensor out_page_indices,
                                         int max_c4_seq_len,
                                         int page_size,
                                         float weight_scale,
                                         bool clean_logits) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    check_hygon_tensor(q, "deepseek_v4_c4_sparse_attn_indexer_");
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    auto q_at = infinicore::adaptor::to_aten_tensor(q);
    auto weights_at = infinicore::adaptor::to_aten_tensor(indexer_weights);
    auto cache_raw_at = infinicore::adaptor::to_aten_tensor(indexer_kv_cache_raw);
    auto c4_seq_lens_at = infinicore::adaptor::to_aten_tensor(c4_seq_lens);
    auto page_table_at = infinicore::adaptor::to_aten_tensor(page_table);
    auto logits_at = infinicore::adaptor::to_aten_tensor(logits);
    auto out_indices_at = infinicore::adaptor::to_aten_tensor(out_page_indices);

    if (q_at.dim() != 3 || q_at.size(2) != kC4IndexerHeadDim) {
        throw std::runtime_error("deepseek_v4_c4_sparse_attn_indexer_ expects q [batch, heads, 128].");
    }
    if (cache_raw_at.dim() != 2 || cache_raw_at.size(1) != page_size * (kC4IndexerHeadDim + kC4IndexerScaleBytes)) {
        throw std::runtime_error("deepseek_v4_c4_sparse_attn_indexer_ expects raw indexer cache [blocks, page_size * 132].");
    }
    if (max_c4_seq_len <= 0) {
        throw std::runtime_error("deepseek_v4_c4_sparse_attn_indexer_ max_c4_seq_len must be positive.");
    }
    if (logits_at.dim() != 2 || logits_at.size(0) < q_at.size(0) || logits_at.size(1) < max_c4_seq_len) {
        throw std::runtime_error("deepseek_v4_c4_sparse_attn_indexer_ logits workspace shape mismatch.");
    }

    auto q_contig = q_at.contiguous();
    auto weights_contig = weights_at.contiguous();
    auto q_quant_native = at::empty_like(q_contig, q_contig.options().dtype(c4_indexer_fp8_dtype()));
    auto q_scale_native = at::empty({q_contig.size(0), q_contig.size(1), 1}, q_contig.options().dtype(at::kFloat));
    auto fused_weights = at::empty_like(weights_contig, weights_contig.options().dtype(at::kFloat));
    deepseek_v4_sparse_attn_indexer::launch_c4_act_quant_fused_scale(
        q_contig.data_ptr(),
        c4_at_scalar_type_for_kernel(q_contig, "deepseek_v4_c4_sparse_attn_indexer_"),
        weights_contig.data_ptr(),
        c4_at_scalar_type_for_kernel(weights_contig, "deepseek_v4_c4_sparse_attn_indexer_"),
        reinterpret_cast<uint8_t *>(q_quant_native.data_ptr()),
        reinterpret_cast<float *>(q_scale_native.data_ptr()),
        reinterpret_cast<float *>(fused_weights.data_ptr()),
        q_contig.size(0) * q_contig.size(1),
        weight_scale,
        current_accelerator_stream());
    auto q_fp8 = q_quant_native.unsqueeze(1);
    auto cache_view = cache_raw_at.view({cache_raw_at.size(0), page_size, 1, kC4IndexerHeadDim + kC4IndexerScaleBytes});
    auto seq_lens_for_gemm = as_c4_seq_lens_for_gemm(c4_seq_lens_at);
    auto seq_lens_flat = as_c4_seq_lens_flat(c4_seq_lens_at);

    std::optional<at::Tensor> schedule_meta = std::nullopt;
    auto paged_fn = lightop_symbols().paged_mqa_logits;
    auto result = paged_fn(q_fp8,
                           cache_view,
                           fused_weights,
                           seq_lens_for_gemm,
                           page_table_at,
                           schedule_meta,
                           max_c4_seq_len,
                           clean_logits);

    auto logits_view = logits_at.slice(0, 0, q_at.size(0)).slice(1, 0, max_c4_seq_len);
    logits_view.copy_(result);
    topk_transform_512_dispatch_at(
        logits_view,
        seq_lens_flat,
        page_table_at,
        out_indices_at,
        page_size,
        "deepseek_v4_c4_sparse_attn_indexer_");
#else
    (void)q;
    (void)indexer_weights;
    (void)indexer_kv_cache_raw;
    (void)c4_seq_lens;
    (void)page_table;
    (void)logits;
    (void)out_page_indices;
    (void)max_c4_seq_len;
    (void)page_size;
    (void)weight_scale;
    (void)clean_logits;
    throw std::runtime_error("deepseek_v4_c4_sparse_attn_indexer_ requires an ATen-enabled HYGON build with lightop.");
#endif
}


void deepseek_v4_c4_sparse_attn_indexer_no_logits_impl(const Tensor &q,
                                                   const Tensor &indexer_weights,
                                                   const Tensor &indexer_kv_cache_raw,
                                                   const Tensor &c4_seq_lens,
                                                   const Tensor &page_table,
                                                   Tensor out_page_indices,
                                                   int max_c4_seq_len,
                                                   int page_size,
                                                   float weight_scale,
                                                   bool clean_logits) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    check_hygon_tensor(q, "deepseek_v4_c4_sparse_attn_indexer_no_logits_");
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    auto q_at = infinicore::adaptor::to_aten_tensor(q);
    auto weights_at = infinicore::adaptor::to_aten_tensor(indexer_weights);
    auto cache_raw_at = infinicore::adaptor::to_aten_tensor(indexer_kv_cache_raw);
    auto c4_seq_lens_at = infinicore::adaptor::to_aten_tensor(c4_seq_lens);
    auto page_table_at = infinicore::adaptor::to_aten_tensor(page_table);
    auto out_indices_at = infinicore::adaptor::to_aten_tensor(out_page_indices);

    if (q_at.dim() != 3 || q_at.size(2) != kC4IndexerHeadDim) {
        throw std::runtime_error("deepseek_v4_c4_sparse_attn_indexer_no_logits_ expects q [batch, heads, 128].");
    }
    if (cache_raw_at.dim() != 2 || cache_raw_at.size(1) != page_size * (kC4IndexerHeadDim + kC4IndexerScaleBytes)) {
        throw std::runtime_error("deepseek_v4_c4_sparse_attn_indexer_no_logits_ expects raw indexer cache [blocks, page_size * 132].");
    }
    if (max_c4_seq_len <= 0) {
        throw std::runtime_error("deepseek_v4_c4_sparse_attn_indexer_no_logits_ max_c4_seq_len must be positive.");
    }

    auto q_contig = q_at.contiguous();
    auto weights_contig = weights_at.contiguous();
    auto q_quant_native = at::empty_like(q_contig, q_contig.options().dtype(c4_indexer_fp8_dtype()));
    auto q_scale_native = at::empty({q_contig.size(0), q_contig.size(1), 1}, q_contig.options().dtype(at::kFloat));
    auto fused_weights = at::empty_like(weights_contig, weights_contig.options().dtype(at::kFloat));
    deepseek_v4_sparse_attn_indexer::launch_c4_act_quant_fused_scale(
        q_contig.data_ptr(),
        c4_at_scalar_type_for_kernel(q_contig, "deepseek_v4_c4_sparse_attn_indexer_no_logits_"),
        weights_contig.data_ptr(),
        c4_at_scalar_type_for_kernel(weights_contig, "deepseek_v4_c4_sparse_attn_indexer_no_logits_"),
        reinterpret_cast<uint8_t *>(q_quant_native.data_ptr()),
        reinterpret_cast<float *>(q_scale_native.data_ptr()),
        reinterpret_cast<float *>(fused_weights.data_ptr()),
        q_contig.size(0) * q_contig.size(1),
        weight_scale,
        current_accelerator_stream());

    auto q_fp8 = q_quant_native.unsqueeze(1);
    auto cache_view = cache_raw_at.view({cache_raw_at.size(0), page_size, 1, kC4IndexerHeadDim + kC4IndexerScaleBytes});
    auto seq_lens_for_gemm = as_c4_seq_lens_for_gemm(c4_seq_lens_at);
    auto seq_lens_flat = as_c4_seq_lens_flat(c4_seq_lens_at);

    std::optional<at::Tensor> schedule_meta = std::nullopt;
    auto paged_fn = lightop_symbols().paged_mqa_logits;
    auto result = paged_fn(q_fp8,
                           cache_view,
                           fused_weights,
                           seq_lens_for_gemm,
                           page_table_at,
                           schedule_meta,
                           max_c4_seq_len,
                           clean_logits);
    auto logits_view = result.contiguous();
    if (logits_view.dim() != 2 || logits_view.size(0) < q_at.size(0) || logits_view.size(1) < max_c4_seq_len) {
        throw std::runtime_error("deepseek_v4_c4_sparse_attn_indexer_no_logits_ paged logits shape mismatch.");
    }
    if (logits_view.size(0) != q_at.size(0) || logits_view.size(1) != max_c4_seq_len) {
        logits_view = logits_view.slice(0, 0, q_at.size(0)).slice(1, 0, max_c4_seq_len).contiguous();
    }
    topk_transform_512_dispatch_at(
        logits_view,
        seq_lens_flat,
        page_table_at,
        out_indices_at,
        page_size,
        "deepseek_v4_c4_sparse_attn_indexer_no_logits_");
#else
    (void)q;
    (void)indexer_weights;
    (void)indexer_kv_cache_raw;
    (void)c4_seq_lens;
    (void)page_table;
    (void)out_page_indices;
    (void)max_c4_seq_len;
    (void)page_size;
    (void)weight_scale;
    (void)clean_logits;
    throw std::runtime_error("deepseek_v4_c4_sparse_attn_indexer_no_logits_ requires an ATen-enabled HYGON build with lightop.");
#endif
}


DeepseekV4C4SparseAttnIndexerNoLogits::DeepseekV4C4SparseAttnIndexerNoLogits(const Tensor &q,
                                                                               const Tensor &indexer_weights,
                                                                               const Tensor &indexer_kv_cache_raw,
                                                                               const Tensor &c4_seq_lens,
                                                                               const Tensor &page_table,
                                                                               Tensor out_page_indices,
                                                                               int max_c4_seq_len,
                                                                               int page_size,
                                                                               float weight_scale,
                                                                               bool clean_logits) {
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(),
                                 q,
                                 indexer_weights,
                                 indexer_kv_cache_raw,
                                 c4_seq_lens,
                                 page_table,
                                 out_page_indices,
                                 max_c4_seq_len,
                                 page_size,
                                 weight_scale,
                                 clean_logits);
}

void DeepseekV4C4SparseAttnIndexerNoLogits::execute(const Tensor &q,
                                                     const Tensor &indexer_weights,
                                                     const Tensor &indexer_kv_cache_raw,
                                                     const Tensor &c4_seq_lens,
                                                     const Tensor &page_table,
                                                     Tensor out_page_indices,
                                                     int max_c4_seq_len,
                                                     int page_size,
                                                     float weight_scale,
                                                     bool clean_logits) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4C4SparseAttnIndexerNoLogits,
                                      q,
                                      indexer_weights,
                                      indexer_kv_cache_raw,
                                      c4_seq_lens,
                                      page_table,
                                      out_page_indices,
                                      max_c4_seq_len,
                                      page_size,
                                      weight_scale,
                                      clean_logits);
}

namespace deepseek_v4_c4_sparse_attn_indexer_no_logits_graph_impl {

struct PlannedMeta {
    graph::GraphTensor q;
    graph::GraphTensor indexer_weights;
    graph::GraphTensor indexer_kv_cache_raw;
    graph::GraphTensor c4_seq_lens;
    graph::GraphTensor page_table;
    graph::GraphTensor out_page_indices;
    int max_c4_seq_len;
    int page_size;
    float weight_scale;
    bool clean_logits;
};

void *plan(const Tensor &q,
           const Tensor &indexer_weights,
           const Tensor &indexer_kv_cache_raw,
           const Tensor &c4_seq_lens,
           const Tensor &page_table,
           Tensor out_page_indices,
           int max_c4_seq_len,
           int page_size,
           float weight_scale,
           bool clean_logits) {
#if defined(ENABLE_HYGON_API)
    if (q->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("DeepseekV4C4SparseAttnIndexerNoLogits expects HYGON tensors in this build.");
    }
#endif
    if (q->ndim() != 3 || q->size(2) != static_cast<size_t>(kC4IndexerHeadDim)) {
        throw std::runtime_error("DeepseekV4C4SparseAttnIndexerNoLogits expects q [batch, heads, 128].");
    }
    if (out_page_indices->ndim() != 2 || out_page_indices->size(0) != q->size(0) || out_page_indices->size(1) < static_cast<size_t>(kC4TopK) || out_page_indices->dtype() != DataType::I32) {
        throw std::runtime_error("DeepseekV4C4SparseAttnIndexerNoLogits expects output page indices [batch, >=512] int32.");
    }
    return new PlannedMeta{graph::GraphTensor(q),
                           graph::GraphTensor(indexer_weights),
                           graph::GraphTensor(indexer_kv_cache_raw),
                           graph::GraphTensor(c4_seq_lens),
                           graph::GraphTensor(page_table),
                           graph::GraphTensor(out_page_indices),
                           max_c4_seq_len,
                           page_size,
                           weight_scale,
                           clean_logits};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_c4_sparse_attn_indexer_no_logits_impl(planned->q,
                                                      planned->indexer_weights,
                                                      planned->indexer_kv_cache_raw,
                                                      planned->c4_seq_lens,
                                                      planned->page_table,
                                                      planned->out_page_indices,
                                                      planned->max_c4_seq_len,
                                                      planned->page_size,
                                                      planned->weight_scale,
                                                      planned->clean_logits);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_c4_sparse_attn_indexer_no_logits_graph_impl

namespace deepseek_v4_c4_sparse_attn_indexer_no_logits_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4C4SparseAttnIndexerNoLogits,
                                       &deepseek_v4_c4_sparse_attn_indexer_no_logits_graph_impl::plan,
                                       &deepseek_v4_c4_sparse_attn_indexer_no_logits_graph_impl::run,
                                       &deepseek_v4_c4_sparse_attn_indexer_no_logits_graph_impl::cleanup);
} // namespace deepseek_v4_c4_sparse_attn_indexer_no_logits_register

void deepseek_v4_c4_sparse_attn_indexer_no_logits_(const Tensor &q,
                                                   const Tensor &indexer_weights,
                                                   const Tensor &indexer_kv_cache_raw,
                                                   const Tensor &c4_seq_lens,
                                                   const Tensor &page_table,
                                                   Tensor out_page_indices,
                                                   int max_c4_seq_len,
                                                   int page_size,
                                                   float weight_scale,
                                                   bool clean_logits) {
    DeepseekV4C4SparseAttnIndexerNoLogits::execute(q,
                                                   indexer_weights,
                                                   indexer_kv_cache_raw,
                                                   c4_seq_lens,
                                                   page_table,
                                                   out_page_indices,
                                                   max_c4_seq_len,
                                                   page_size,
                                                   weight_scale,
                                                   clean_logits);
}

} // namespace infinicore::op
