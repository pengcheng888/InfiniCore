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

void deepseek_v4_silu_and_mul_dispatcher_(Tensor out, const Tensor &x) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (x->device().getType() == Device::Type::HYGON) {
        c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
        auto out_at = infinicore::adaptor::to_aten_tensor(out);
        auto x_at = infinicore::adaptor::to_aten_tensor(x);
        // 下面是第一版代码。经过完善后，才移植了kernel，形成了第二版代码。
        static auto op = c10::Dispatcher::singleton()
                             .findSchemaOrThrow("sgl_kernel::silu_and_mul", "")
                             .typed<void(at::Tensor &, at::Tensor &)>();
        op.call(out_at, x_at);
        return;
    }
#else
    if (x->device().getType() == Device::Type::NVIDIA) {
        c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
        auto out_at = infinicore::adaptor::to_aten_tensor(out);
        auto x_at = infinicore::adaptor::to_aten_tensor(x);
        // 下面是第一版代码。经过完善后，才移植了kernel，形成了第二版代码。
        static auto op = c10::Dispatcher::singleton()
                             .findSchemaOrThrow("sgl_kernel::silu_and_mul", "")
                             .typed<void(at::Tensor &, at::Tensor &)>();
        op.call(out_at, x_at);
        return;
    }
#endif
#endif
    (void)out;
    (void)x;
    throw std::runtime_error("deepseek_v4_silu_and_mul_dispatcher_ requires an ATen-enabled HYGON/NVIDIA build.");
}

} // namespace infinicore::op
