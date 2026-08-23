#include "infinicore/ops/deepseek_v4_paged_mqa_logits.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
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
constexpr const char *kMetadataSymbol = "_ZN8deepgemm29get_paged_mqa_logits_metadataERKN2at6TensorEii";
constexpr const char *kLogitsSymbol = "_ZN8deepgemm16paged_mqa_logitsERKN2at6TensorES3_S3_S3_S3_RSt8optionalIS1_ERKiRKb";

using MetadataFn = at::Tensor (*)(const at::Tensor &, int, int);
using LogitsFn = at::Tensor (*)(const at::Tensor &,
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
        throw std::runtime_error(std::string("deepgemm SO is missing required symbol ") + symbol +
                                 (error != nullptr ? std::string(": ") + error : ""));
    }
    return reinterpret_cast<Fn>(fn);
}

void *open_deepgemm_so() {
    std::vector<std::string> candidates;
    if (const char *env_path = std::getenv("INFINICORE_DEEPGEMM_OP_SO")) {
        if (env_path[0] != '\0') {
            candidates.emplace_back(env_path);
        }
    }
    candidates.emplace_back("/usr/local/lib/python3.10/dist-packages/deepgemm/op.cpython-310-x86_64-linux-gnu.so");
    candidates.emplace_back("/usr/local/lib/python3.11/dist-packages/deepgemm/op.cpython-311-x86_64-linux-gnu.so");
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
    throw std::runtime_error("failed to load deepgemm op SO. Set INFINICORE_DEEPGEMM_OP_SO to deepgemm/op*.so." + errors.str());
}

struct DeepgemmSymbols {
    void *handle{nullptr};
    MetadataFn metadata{nullptr};
    LogitsFn logits{nullptr};
};

const DeepgemmSymbols &deepgemm_symbols() {
    static DeepgemmSymbols symbols;
    static std::once_flag once;
    std::call_once(once, [] {
        symbols.handle = open_deepgemm_so();
        symbols.metadata = checked_symbol<MetadataFn>(symbols.handle, kMetadataSymbol);
        symbols.logits = checked_symbol<LogitsFn>(symbols.handle, kLogitsSymbol);
    });
    return symbols;
}
#endif

} // namespace

void deepseek_v4_paged_mqa_logits_metadata_(const Tensor &context_lens,
                                            Tensor schedule_meta,
                                            int block_kv,
                                            int num_sms) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    check_hygon_tensor(context_lens, "deepseek_v4_paged_mqa_logits_metadata_");
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    auto context_lens_at = infinicore::adaptor::to_aten_tensor(context_lens);
    auto schedule_meta_at = infinicore::adaptor::to_aten_tensor(schedule_meta);

    auto fn = deepgemm_symbols().metadata;
    auto result = fn(context_lens_at, block_kv, num_sms);
    schedule_meta_at.copy_(result);
#else
    (void)context_lens;
    (void)schedule_meta;
    (void)block_kv;
    (void)num_sms;
    throw std::runtime_error("deepseek_v4_paged_mqa_logits_metadata_ requires an ATen-enabled HYGON build with deepgemm.");
#endif
}

void deepseek_v4_paged_mqa_logits_(const Tensor &q,
                                   const Tensor &fused_kv_cache,
                                   const Tensor &weights,
                                   const Tensor &context_lens,
                                   const Tensor &block_table,
                                   const Tensor &schedule_meta,
                                   Tensor logits,
                                   int max_context_len,
                                   bool clean_logits) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    check_hygon_tensor(q, "deepseek_v4_paged_mqa_logits_");
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    auto q_at = infinicore::adaptor::to_aten_tensor(q);
    auto fused_kv_cache_at = infinicore::adaptor::to_aten_tensor(fused_kv_cache);
    auto weights_at = infinicore::adaptor::to_aten_tensor(weights);
    auto context_lens_at = infinicore::adaptor::to_aten_tensor(context_lens);
    auto block_table_at = infinicore::adaptor::to_aten_tensor(block_table);
    auto schedule_meta_tensor = infinicore::adaptor::to_aten_tensor(schedule_meta);
    auto logits_at = infinicore::adaptor::to_aten_tensor(logits);
    std::optional<at::Tensor> schedule_meta_at = schedule_meta_tensor;

    auto fn = deepgemm_symbols().logits;
    auto result = fn(q_at,
                     fused_kv_cache_at,
                     weights_at,
                     context_lens_at,
                     block_table_at,
                     schedule_meta_at,
                     max_context_len,
                     clean_logits);
    logits_at.copy_(result);
#else
    (void)q;
    (void)fused_kv_cache;
    (void)weights;
    (void)context_lens;
    (void)block_table;
    (void)schedule_meta;
    (void)logits;
    (void)max_context_len;
    (void)clean_logits;
    throw std::runtime_error("deepseek_v4_paged_mqa_logits_ requires an ATen-enabled HYGON build with deepgemm.");
#endif
}

} // namespace infinicore::op
