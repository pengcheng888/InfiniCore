#include "infinicore/ops/deepseek_v4_rotary_embedding.hpp"

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

void deepseek_v4_rotary_embedding_(const Tensor &positions,
                                   Tensor query,
                                   std::optional<Tensor> key,
                                   int head_size,
                                   const Tensor &cos_sin_cache,
                                   bool is_neox) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (query->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_rotary_embedding_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (query->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_rotary_embedding_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto positions_at = infinicore::adaptor::to_aten_tensor(positions);
    auto query_at = infinicore::adaptor::to_aten_tensor(query);
    auto cos_sin_at = infinicore::adaptor::to_aten_tensor(cos_sin_cache);
    std::optional<at::Tensor> key_at = std::nullopt;
    if (key.has_value()) {
        key_at = infinicore::adaptor::to_aten_tensor(*key);
    }

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::rotary_embedding", "")
                         .typed<void(at::Tensor &, at::Tensor &, std::optional<at::Tensor>, int64_t, at::Tensor &, bool)>();
    op.call(positions_at, query_at, key_at, static_cast<int64_t>(head_size), cos_sin_at, is_neox);
#else
    (void)positions;
    (void)query;
    (void)key;
    (void)head_size;
    (void)cos_sin_cache;
    (void)is_neox;
    throw std::runtime_error("deepseek_v4_rotary_embedding_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
