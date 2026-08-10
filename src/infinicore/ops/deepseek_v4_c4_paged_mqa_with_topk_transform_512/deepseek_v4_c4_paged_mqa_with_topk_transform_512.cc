#include "infinicore/ops/deepseek_v4_c4_paged_mqa_with_topk_transform_512.hpp"

#include "../deepseek_v4_topk_transform_512/deepseek_v4_topk_transform_512_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#endif
#endif

#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4C4PagedMqaWithTopkTransform512);

namespace {

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
constexpr int64_t kC4IndexerHeadDim = 128;
constexpr int64_t kC4IndexerScaleBytes = 4;
constexpr int64_t kC4TopK = 512;
constexpr const char *kLightopPagedMqaLogitsSymbol = "_ZN7lightop9deep_gemm16paged_mqa_logitsERKN2at6TensorES4_S4_S4_S4_RSt8optionalIS2_ERKiRKb";

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
        throw std::runtime_error(std::string("lightop SO is missing required symbol ") + symbol + (error != nullptr ? std::string(": ") + error : ""));
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
    PagedMqaLogitsFn paged_mqa_logits{nullptr};
};

const LightopSymbols &lightop_symbols() {
    static LightopSymbols symbols;
    static std::once_flag once;
    std::call_once(once, [] {
        symbols.handle = open_lightop_so();
        symbols.paged_mqa_logits = checked_symbol<PagedMqaLogitsFn>(symbols.handle, kLightopPagedMqaLogitsSymbol);
    });
    return symbols;
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
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ expects c4_seq_lens [batch] or [batch, 1].");
    }
    return flat.to(at::kInt).unsqueeze(1);
}

at::Tensor c4_fp8_e4m3fn_view(const Tensor &tensor) {
    auto sizes = std::vector<int64_t>(tensor->shape().begin(), tensor->shape().end());
    auto strides = tensor->strides();
    auto options = at::TensorOptions()
                       .dtype(at::ScalarType::Float8_e4m3fn)
                       .device(infinicore::adaptor::to_at_device(tensor->device()))
                       .requires_grad(false);
    auto deleter = [](void *) {};
    return at::from_blob((void *)(tensor->data()), sizes, strides, deleter, options);
}

void topk_transform_512_at(const at::Tensor &scores,
                           const at::Tensor &seq_lens,
                           const at::Tensor &page_tables,
                           at::Tensor &out_page_indices,
                           int page_size) {
    if (scores.dim() != 2 || page_tables.dim() != 2 || out_page_indices.dim() != 2) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ expects scores/page_tables/out_page_indices to be 2-D.");
    }
    const int64_t batch = scores.size(0);
    const int64_t max_seq_len = scores.size(1);
    if (seq_lens.dim() != 1 || seq_lens.size(0) != batch || page_tables.size(0) != batch || out_page_indices.size(0) != batch || out_page_indices.size(1) < kC4TopK) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ topk shape mismatch.");
    }
    if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ page_size must be a positive power of two.");
    }

    auto int_opts = out_page_indices.options().dtype(at::kInt);
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
    deepseek_v4_topk_transform_512::launch_topk_transform_512(
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
        context::getStream());
}

void deepseek_v4_c4_paged_mqa_with_topk_transform_512_impl(const Tensor &q_fp8,
                                                           const Tensor &fused_weights,
                                                           const Tensor &indexer_kv_cache_raw,
                                                           const Tensor &c4_seq_lens,
                                                           const Tensor &page_table,
                                                           Tensor out_page_indices,
                                                           int max_c4_seq_len,
                                                           int page_size,
                                                           bool clean_logits) {
    constexpr const char *op_name = "deepseek_v4_c4_paged_mqa_with_topk_transform_512_";
    check_hygon_tensor(q_fp8, op_name);
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    auto q_at = c4_fp8_e4m3fn_view(q_fp8);
    auto weights_at = infinicore::adaptor::to_aten_tensor(fused_weights);
    auto cache_raw_at = infinicore::adaptor::to_aten_tensor(indexer_kv_cache_raw);
    auto c4_seq_lens_at = infinicore::adaptor::to_aten_tensor(c4_seq_lens);
    auto page_table_at = infinicore::adaptor::to_aten_tensor(page_table);
    auto out_page_indices_at = infinicore::adaptor::to_aten_tensor(out_page_indices);

    if (q_at.dim() != 3 || q_at.size(2) != kC4IndexerHeadDim) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ expects q_fp8 [batch, heads, 128].");
    }
    if (weights_at.dim() != 2 || weights_at.size(0) != q_at.size(0) || weights_at.size(1) != q_at.size(1) || weights_at.scalar_type() != at::kFloat) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ expects fused_weights [batch, heads] fp32.");
    }
    if (cache_raw_at.dim() != 2 || cache_raw_at.size(1) != page_size * (kC4IndexerHeadDim + kC4IndexerScaleBytes)) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ expects raw indexer cache [blocks, page_size * 132].");
    }
    auto c4_seq_lens_flat = squeeze_last_if_needed(c4_seq_lens_at);
    if (c4_seq_lens_flat.dim() != 1 || c4_seq_lens_flat.size(0) != q_at.size(0)) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ expects c4_seq_lens [batch] or [batch, 1].");
    }
    if (page_table_at.dim() != 2 || page_table_at.size(0) != q_at.size(0)) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ expects page_table [batch, pages].");
    }
    if (out_page_indices_at.dim() != 2 || out_page_indices_at.size(0) != q_at.size(0) || out_page_indices_at.size(1) < kC4TopK || out_page_indices_at.scalar_type() != at::kInt) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ expects out_page_indices [batch, >=512] int32.");
    }
    if (max_c4_seq_len <= 0) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ max_c4_seq_len must be positive.");
    }
    if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ page_size must be a positive power of two.");
    }
    if (!q_fp8->is_contiguous() || !fused_weights->is_contiguous() || !indexer_kv_cache_raw->is_contiguous() || !c4_seq_lens->is_contiguous() || !page_table->is_contiguous() || !out_page_indices->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ expects contiguous tensors.");
    }

    auto q_for_gemm = q_at.unsqueeze(1);
    auto cache_view = cache_raw_at.view({cache_raw_at.size(0), page_size, 1, kC4IndexerHeadDim + kC4IndexerScaleBytes});
    auto seq_lens_for_gemm = as_c4_seq_lens_for_gemm(c4_seq_lens_at);
    std::optional<at::Tensor> schedule_meta = std::nullopt;
    auto result = lightop_symbols().paged_mqa_logits(q_for_gemm,
                                                     cache_view,
                                                     weights_at,
                                                     seq_lens_for_gemm,
                                                     page_table_at,
                                                     schedule_meta,
                                                     max_c4_seq_len,
                                                     clean_logits);
    topk_transform_512_dispatch_at(result, c4_seq_lens_flat, page_table_at, out_page_indices_at, page_size, op_name);
}
#endif

} // namespace

DeepseekV4C4PagedMqaWithTopkTransform512::DeepseekV4C4PagedMqaWithTopkTransform512(const Tensor &q_fp8,
                                                                                   const Tensor &fused_weights,
                                                                                   const Tensor &indexer_kv_cache_raw,
                                                                                   const Tensor &c4_seq_lens,
                                                                                   const Tensor &page_table,
                                                                                   Tensor out_page_indices,
                                                                                   int max_c4_seq_len,
                                                                                   int page_size,
                                                                                   bool clean_logits) {
    INFINICORE_GRAPH_OP_DISPATCH(q_fp8->device().getType(),
                                 q_fp8,
                                 fused_weights,
                                 indexer_kv_cache_raw,
                                 c4_seq_lens,
                                 page_table,
                                 out_page_indices,
                                 max_c4_seq_len,
                                 page_size,
                                 clean_logits);
}

void DeepseekV4C4PagedMqaWithTopkTransform512::execute(const Tensor &q_fp8,
                                                       const Tensor &fused_weights,
                                                       const Tensor &indexer_kv_cache_raw,
                                                       const Tensor &c4_seq_lens,
                                                       const Tensor &page_table,
                                                       Tensor out_page_indices,
                                                       int max_c4_seq_len,
                                                       int page_size,
                                                       bool clean_logits) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4C4PagedMqaWithTopkTransform512,
                                      q_fp8,
                                      fused_weights,
                                      indexer_kv_cache_raw,
                                      c4_seq_lens,
                                      page_table,
                                      out_page_indices,
                                      max_c4_seq_len,
                                      page_size,
                                      clean_logits);
}

namespace deepseek_v4_c4_paged_mqa_with_topk_transform_512_graph_impl {

struct PlannedMeta {
    graph::GraphTensor q_fp8;
    graph::GraphTensor fused_weights;
    graph::GraphTensor indexer_kv_cache_raw;
    graph::GraphTensor c4_seq_lens;
    graph::GraphTensor page_table;
    graph::GraphTensor out_page_indices;
    int max_c4_seq_len;
    int page_size;
    bool clean_logits;
};

void *plan(const Tensor &q_fp8,
           const Tensor &fused_weights,
           const Tensor &indexer_kv_cache_raw,
           const Tensor &c4_seq_lens,
           const Tensor &page_table,
           Tensor out_page_indices,
           int max_c4_seq_len,
           int page_size,
           bool clean_logits) {
    return new PlannedMeta{graph::GraphTensor(q_fp8),
                           graph::GraphTensor(fused_weights),
                           graph::GraphTensor(indexer_kv_cache_raw),
                           graph::GraphTensor(c4_seq_lens),
                           graph::GraphTensor(page_table),
                           graph::GraphTensor(out_page_indices),
                           max_c4_seq_len,
                           page_size,
                           clean_logits};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    deepseek_v4_c4_paged_mqa_with_topk_transform_512_impl(planned->q_fp8,
                                                          planned->fused_weights,
                                                          planned->indexer_kv_cache_raw,
                                                          planned->c4_seq_lens,
                                                          planned->page_table,
                                                          planned->out_page_indices,
                                                          planned->max_c4_seq_len,
                                                          planned->page_size,
                                                          planned->clean_logits);
#else
    (void)planned;
    throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ requires an ATen-enabled HYGON build with lightop.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_c4_paged_mqa_with_topk_transform_512_graph_impl

namespace deepseek_v4_c4_paged_mqa_with_topk_transform_512_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4C4PagedMqaWithTopkTransform512,
                                       &deepseek_v4_c4_paged_mqa_with_topk_transform_512_graph_impl::plan,
                                       &deepseek_v4_c4_paged_mqa_with_topk_transform_512_graph_impl::run,
                                       &deepseek_v4_c4_paged_mqa_with_topk_transform_512_graph_impl::cleanup);
} // namespace deepseek_v4_c4_paged_mqa_with_topk_transform_512_register

void deepseek_v4_c4_paged_mqa_with_topk_transform_512_(const Tensor &q_fp8,
                                                       const Tensor &fused_weights,
                                                       const Tensor &indexer_kv_cache_raw,
                                                       const Tensor &c4_seq_lens,
                                                       const Tensor &page_table,
                                                       Tensor out_page_indices,
                                                       int max_c4_seq_len,
                                                       int page_size,
                                                       bool clean_logits) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    DeepseekV4C4PagedMqaWithTopkTransform512::execute(q_fp8,
                                                      fused_weights,
                                                      indexer_kv_cache_raw,
                                                      c4_seq_lens,
                                                      page_table,
                                                      out_page_indices,
                                                      max_c4_seq_len,
                                                      page_size,
                                                      clean_logits);
#else
    (void)q_fp8;
    (void)fused_weights;
    (void)indexer_kv_cache_raw;
    (void)c4_seq_lens;
    (void)page_table;
    (void)out_page_indices;
    (void)max_c4_seq_len;
    (void)page_size;
    (void)clean_logits;
    throw std::runtime_error("deepseek_v4_c4_paged_mqa_with_topk_transform_512_ requires an ATen-enabled HYGON build with lightop.");
#endif
}

} // namespace infinicore::op
