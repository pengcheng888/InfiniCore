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

#include <dlfcn.h>
#include <optional>
#include <stdexcept>
#include <string>

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

void *resolve_deepgemm_symbol(const char *symbol) {
    void *fn = dlsym(RTLD_DEFAULT, symbol);
    if (fn == nullptr) {
        throw std::runtime_error(
            std::string("deepseek_v4_paged_mqa_logits requires deepgemm.op to be loaded with RTLD_GLOBAL; missing symbol: ") + symbol);
    }
    return fn;
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

    static auto fn = reinterpret_cast<MetadataFn>(resolve_deepgemm_symbol(kMetadataSymbol));
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

    static auto fn = reinterpret_cast<LogitsFn>(resolve_deepgemm_symbol(kLogitsSymbol));
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
