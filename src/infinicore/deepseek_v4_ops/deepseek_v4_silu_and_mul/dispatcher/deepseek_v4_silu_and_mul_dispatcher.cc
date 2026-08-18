#include "infinicore/ops/deepseek_v4_silu_and_mul.hpp"

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

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
namespace {

auto &sgl_silu_and_mul_op() {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::silu_and_mul", "")
                         .typed<void(at::Tensor &, at::Tensor &)>();
    return op;
}

} // namespace
#endif

void deepseek_v4_silu_and_mul_dispatcher_(Tensor out, const Tensor &x) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (x->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_silu_and_mul_dispatcher_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (x->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_silu_and_mul_dispatcher_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto out_at = infinicore::adaptor::to_aten_tensor(out);
    auto x_at = infinicore::adaptor::to_aten_tensor(x);
    sgl_silu_and_mul_op().call(out_at, x_at);
    return;
#endif
    (void)out;
    (void)x;
    throw std::runtime_error("deepseek_v4_silu_and_mul_dispatcher_ requires an ATen-enabled HYGON/NVIDIA build.");
}

} // namespace infinicore::op
