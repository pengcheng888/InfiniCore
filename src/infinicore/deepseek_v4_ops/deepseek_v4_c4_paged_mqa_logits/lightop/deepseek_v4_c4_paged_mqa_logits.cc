#include "infinicore/ops/deepseek_v4_c4_paged_mqa_logits.hpp"

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
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op {

namespace {

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
constexpr int64_t kC4IndexerHeadDim = 128;
constexpr int64_t kC4IndexerScaleBytes = 4;
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
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_logits_ expects c4_seq_lens [batch] or [batch, 1].");
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

void run_c4_paged_mqa_logits_lightop(const Tensor &q_fp8,
                                     const Tensor &fused_weights,
                                     const Tensor &indexer_kv_cache_raw,
                                     const Tensor &c4_seq_lens,
                                     const Tensor &page_table,
                                     Tensor logits,
                                     int max_c4_seq_len,
                                     int page_size,
                                     bool clean_logits) {
    constexpr const char *op_name = "deepseek_v4_c4_paged_mqa_logits_";
    check_hygon_tensor(q_fp8, op_name);
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    auto q_at = c4_fp8_e4m3fn_view(q_fp8);
    auto weights_at = infinicore::adaptor::to_aten_tensor(fused_weights);
    auto cache_raw_at = infinicore::adaptor::to_aten_tensor(indexer_kv_cache_raw);
    auto c4_seq_lens_at = infinicore::adaptor::to_aten_tensor(c4_seq_lens);
    auto page_table_at = infinicore::adaptor::to_aten_tensor(page_table);
    auto logits_at = infinicore::adaptor::to_aten_tensor(logits);

    if (q_at.dim() != 3 || q_at.size(2) != kC4IndexerHeadDim) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_logits_ expects q_fp8 [batch, heads, 128].");
    }
    if (weights_at.dim() != 2 || weights_at.size(0) != q_at.size(0) || weights_at.size(1) != q_at.size(1) || weights_at.scalar_type() != at::kFloat) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_logits_ expects fused_weights [batch, heads] fp32.");
    }
    if (cache_raw_at.dim() != 2 || cache_raw_at.size(1) != page_size * (kC4IndexerHeadDim + kC4IndexerScaleBytes)) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_logits_ expects raw indexer cache [blocks, page_size * 132].");
    }
    if (logits_at.dim() != 2 || logits_at.size(0) < q_at.size(0) || logits_at.size(1) < max_c4_seq_len || logits_at.scalar_type() != at::kFloat) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_logits_ logits workspace shape mismatch.");
    }
    if (max_c4_seq_len <= 0) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_logits_ max_c4_seq_len must be positive.");
    }
    if (!q_fp8->is_contiguous() || !fused_weights->is_contiguous() || !indexer_kv_cache_raw->is_contiguous() || !c4_seq_lens->is_contiguous() || !page_table->is_contiguous() || !logits->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_c4_paged_mqa_logits_ expects contiguous tensors.");
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
    auto logits_view = logits_at.slice(0, 0, q_at.size(0)).slice(1, 0, max_c4_seq_len);
    logits_view.copy_(result);
}
#endif

} // namespace

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4C4PagedMqaLogits);

DeepseekV4C4PagedMqaLogits::DeepseekV4C4PagedMqaLogits(const Tensor &q_fp8,
                                                       const Tensor &fused_weights,
                                                       const Tensor &indexer_kv_cache_raw,
                                                       const Tensor &c4_seq_lens,
                                                       const Tensor &page_table,
                                                       Tensor logits,
                                                       int max_c4_seq_len,
                                                       int page_size,
                                                       bool clean_logits) {
    INFINICORE_GRAPH_OP_DISPATCH(q_fp8->device().getType(),
                                 q_fp8,
                                 fused_weights,
                                 indexer_kv_cache_raw,
                                 c4_seq_lens,
                                 page_table,
                                 logits,
                                 max_c4_seq_len,
                                 page_size,
                                 clean_logits);
}

void DeepseekV4C4PagedMqaLogits::execute(const Tensor &q_fp8,
                                         const Tensor &fused_weights,
                                         const Tensor &indexer_kv_cache_raw,
                                         const Tensor &c4_seq_lens,
                                         const Tensor &page_table,
                                         Tensor logits,
                                         int max_c4_seq_len,
                                         int page_size,
                                         bool clean_logits) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4C4PagedMqaLogits,
                                      q_fp8,
                                      fused_weights,
                                      indexer_kv_cache_raw,
                                      c4_seq_lens,
                                      page_table,
                                      logits,
                                      max_c4_seq_len,
                                      page_size,
                                      clean_logits);
}

namespace deepseek_v4_c4_paged_mqa_logits_impl {

struct PlannedMeta {
    graph::GraphTensor q_fp8;
    graph::GraphTensor fused_weights;
    graph::GraphTensor indexer_kv_cache_raw;
    graph::GraphTensor c4_seq_lens;
    graph::GraphTensor page_table;
    graph::GraphTensor logits;
    int max_c4_seq_len;
    int page_size;
    bool clean_logits;
};

void *plan(const Tensor &q_fp8,
           const Tensor &fused_weights,
           const Tensor &indexer_kv_cache_raw,
           const Tensor &c4_seq_lens,
           const Tensor &page_table,
           Tensor logits,
           int max_c4_seq_len,
           int page_size,
           bool clean_logits) {
    return new PlannedMeta{graph::GraphTensor(q_fp8),
                           graph::GraphTensor(fused_weights),
                           graph::GraphTensor(indexer_kv_cache_raw),
                           graph::GraphTensor(c4_seq_lens),
                           graph::GraphTensor(page_table),
                           graph::GraphTensor(logits),
                           max_c4_seq_len,
                           page_size,
                           clean_logits};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    run_c4_paged_mqa_logits_lightop(planned->q_fp8,
                                    planned->fused_weights,
                                    planned->indexer_kv_cache_raw,
                                    planned->c4_seq_lens,
                                    planned->page_table,
                                    planned->logits,
                                    planned->max_c4_seq_len,
                                    planned->page_size,
                                    planned->clean_logits);
#else
    (void)planned;
    throw std::runtime_error("deepseek_v4_c4_paged_mqa_logits_ requires an ATen-enabled HYGON build with lightop.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_c4_paged_mqa_logits_impl

namespace deepseek_v4_c4_paged_mqa_logits_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4C4PagedMqaLogits,
                                       &deepseek_v4_c4_paged_mqa_logits_impl::plan,
                                       &deepseek_v4_c4_paged_mqa_logits_impl::run,
                                       &deepseek_v4_c4_paged_mqa_logits_impl::cleanup);
} // namespace deepseek_v4_c4_paged_mqa_logits_register

} // namespace deepseek_v4

void deepseek_v4_c4_paged_mqa_logits_lightop_(const Tensor &q_fp8,
                                              const Tensor &fused_weights,
                                              const Tensor &indexer_kv_cache_raw,
                                              const Tensor &c4_seq_lens,
                                              const Tensor &page_table,
                                              Tensor logits,
                                              int max_c4_seq_len,
                                              int page_size,
                                              bool clean_logits) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    deepseek_v4::DeepseekV4C4PagedMqaLogits::execute(q_fp8,
                                                     fused_weights,
                                                     indexer_kv_cache_raw,
                                                     c4_seq_lens,
                                                     page_table,
                                                     logits,
                                                     max_c4_seq_len,
                                                     page_size,
                                                     clean_logits);
#else
    (void)q_fp8;
    (void)fused_weights;
    (void)indexer_kv_cache_raw;
    (void)c4_seq_lens;
    (void)page_table;
    (void)logits;
    (void)max_c4_seq_len;
    (void)page_size;
    (void)clean_logits;
    throw std::runtime_error("deepseek_v4_c4_paged_mqa_logits_ requires an ATen-enabled HYGON build with lightop.");
#endif
}

} // namespace infinicore::op
