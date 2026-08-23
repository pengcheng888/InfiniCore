#include "infinicore/ops/deepseek_v4_dcu_cache_alloc.hpp"

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

void deepseek_v4_dcu_alloc_decode_kernel_(const Tensor &seq_lens,
                                          const Tensor &last_loc,
                                          const Tensor &free_page,
                                          Tensor out_indices,
                                          int bs,
                                          int page_size) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (seq_lens->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_dcu_alloc_decode_kernel_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (seq_lens->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_dcu_alloc_decode_kernel_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto seq_lens_at = infinicore::adaptor::to_aten_tensor(seq_lens);
    auto last_loc_at = infinicore::adaptor::to_aten_tensor(last_loc);
    auto free_page_at = infinicore::adaptor::to_aten_tensor(free_page);
    auto out_indices_at = infinicore::adaptor::to_aten_tensor(out_indices);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::dcu_alloc_decode_kernel", "")
                         .typed<void(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t)>();
    op.call(seq_lens_at,
            last_loc_at,
            free_page_at,
            out_indices_at,
            static_cast<int64_t>(bs),
            static_cast<int64_t>(page_size));
#else
    (void)seq_lens;
    (void)last_loc;
    (void)free_page;
    (void)out_indices;
    (void)bs;
    (void)page_size;
    throw std::runtime_error("deepseek_v4_dcu_alloc_decode_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_dcu_alloc_extend_kernel_(const Tensor &pre_lens,
                                          const Tensor &seq_lens,
                                          const Tensor &last_loc,
                                          const Tensor &free_page,
                                          Tensor out_indices,
                                          int bs,
                                          int page_size) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (pre_lens->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_dcu_alloc_extend_kernel_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (pre_lens->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_dcu_alloc_extend_kernel_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto pre_lens_at = infinicore::adaptor::to_aten_tensor(pre_lens);
    auto seq_lens_at = infinicore::adaptor::to_aten_tensor(seq_lens);
    auto last_loc_at = infinicore::adaptor::to_aten_tensor(last_loc);
    auto free_page_at = infinicore::adaptor::to_aten_tensor(free_page);
    auto out_indices_at = infinicore::adaptor::to_aten_tensor(out_indices);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("sgl_kernel::dcu_alloc_extend_kernel", "")
                         .typed<void(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t)>();
    op.call(pre_lens_at,
            seq_lens_at,
            last_loc_at,
            free_page_at,
            out_indices_at,
            static_cast<int64_t>(bs),
            static_cast<int64_t>(page_size));
#else
    (void)pre_lens;
    (void)seq_lens;
    (void)last_loc;
    (void)free_page;
    (void)out_indices;
    (void)bs;
    (void)page_size;
    throw std::runtime_error("deepseek_v4_dcu_alloc_extend_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
