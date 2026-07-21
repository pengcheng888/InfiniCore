#include "infinicore/ops/deepseek_v4_concat_and_cache_mla.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/core/dispatch/Dispatcher.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <stdexcept>

namespace infinicore::op {

void deepseek_v4_concat_and_cache_mla_(const Tensor &kv_c,
                                       const Tensor &k_pe,
                                       Tensor kv_cache,
                                       const Tensor &slot_mapping,
                                       const std::string &kv_cache_dtype,
                                       const Tensor &scale) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (kv_c->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_concat_and_cache_mla_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (kv_c->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_concat_and_cache_mla_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto kv_c_at = infinicore::adaptor::to_aten_tensor(kv_c);
    auto k_pe_at = infinicore::adaptor::to_aten_tensor(k_pe);
    auto kv_cache_at = infinicore::adaptor::to_aten_tensor(kv_cache);
    auto slot_mapping_at = infinicore::adaptor::to_aten_tensor(slot_mapping);
    auto scale_at = infinicore::adaptor::to_aten_tensor(scale);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("_C_cache_ops::concat_and_cache_mla", "")
                         .typed<void(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, const std::string &, at::Tensor &)>();
    op.call(kv_c_at, k_pe_at, kv_cache_at, slot_mapping_at, kv_cache_dtype, scale_at);
#else
    (void)kv_c;
    (void)k_pe;
    (void)kv_cache;
    (void)slot_mapping;
    (void)kv_cache_dtype;
    (void)scale;
    throw std::runtime_error("deepseek_v4_concat_and_cache_mla_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
