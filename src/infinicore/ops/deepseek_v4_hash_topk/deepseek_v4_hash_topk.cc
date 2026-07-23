#include "infinicore/ops/deepseek_v4_hash_topk.hpp"

#include "deepseek_v4_hash_topk_kernel.hpp"

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

#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace {

void check_accelerator_tensor(const Tensor &tensor, const char *op_name) {
#if defined(ENABLE_HYGON_API)
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
#elif defined(ENABLE_NVIDIA_API)
    if (tensor->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error(std::string(op_name) + " expects NVIDIA tensors in this build.");
    }
#else
    (void)tensor;
    (void)op_name;
#endif
}

void check_shapes(const Tensor &topk_weights,
                  const Tensor &topk_indices,
                  const Tensor &router_logits,
                  const Tensor &input_ids,
                  const Tensor &tid2eid) {
    if (router_logits->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_hash_topk_naive_ expects router_logits to be 2-D.");
    }
    if (input_ids->ndim() != 1) {
        throw std::runtime_error("deepseek_v4_hash_topk_naive_ expects input_ids to be 1-D.");
    }
    if (tid2eid->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_hash_topk_naive_ expects tid2eid to be 2-D.");
    }
    if (topk_weights->shape() != topk_indices->shape()) {
        throw std::runtime_error("deepseek_v4_hash_topk_naive_ topk weight/index shape mismatch.");
    }
    if (topk_weights->shape() != Shape{router_logits->size(0), tid2eid->size(1)}) {
        throw std::runtime_error("deepseek_v4_hash_topk_naive_ output shape mismatch.");
    }
    if (input_ids->size(0) != router_logits->size(0)) {
        throw std::runtime_error("deepseek_v4_hash_topk_naive_ input_ids/router token count mismatch.");
    }
}


#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
void check_kernel_tensors(const Tensor &topk_weights,
                          const Tensor &topk_indices,
                          const Tensor &router_logits,
                          const Tensor &input_ids,
                          const Tensor &tid2eid) {
    if (topk_weights->dtype() != DataType::F32 || topk_indices->dtype() != DataType::I32 ||
        router_logits->dtype() != DataType::F32 || input_ids->dtype() != DataType::I64 ||
        (tid2eid->dtype() != DataType::I64 && tid2eid->dtype() != DataType::I32)) {
        throw std::runtime_error("deepseek_v4_hash_topk_kernel_ expects F32 weights/logits, I32 indices, I64 input_ids, and I64/I32 tid2eid.");
    }
    if (!topk_weights->is_contiguous() || !topk_indices->is_contiguous() ||
        !router_logits->is_contiguous() || !input_ids->is_contiguous() || !tid2eid->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_hash_topk_kernel_ expects contiguous tensors.");
    }
}

void *current_accelerator_stream() {
#if defined(ENABLE_HYGON_API)
    return reinterpret_cast<void *>(infinicore::adaptor::get_hip_stream().stream());
#else
    return reinterpret_cast<void *>(infinicore::adaptor::get_cuda_stream().stream());
#endif
}
#endif

} // namespace

void deepseek_v4_hash_topk_naive_(Tensor topk_weights,
                            Tensor topk_indices,
                            const Tensor &router_logits,
                            const Tensor &input_ids,
                            const Tensor &tid2eid,
                            bool renormalize) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(router_logits, "deepseek_v4_hash_topk_naive_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    check_shapes(topk_weights, topk_indices, router_logits, input_ids, tid2eid);

    auto weights_at = infinicore::adaptor::to_aten_tensor(topk_weights);
    auto indices_at = infinicore::adaptor::to_aten_tensor(topk_indices);
    auto logits_at = infinicore::adaptor::to_aten_tensor(router_logits);
    auto input_ids_at = infinicore::adaptor::to_aten_tensor(input_ids).to(at::kLong);
    auto tid2eid_at = infinicore::adaptor::to_aten_tensor(tid2eid).to(at::kLong);

    auto selected = tid2eid_at.index_select(0, input_ids_at);
    auto scores = at::sqrt(at::softplus(logits_at.to(at::kFloat)));
    auto gathered = scores.gather(1, selected);
    if (renormalize) {
        gathered = gathered / gathered.sum(-1, true);
    }
    weights_at.copy_(gathered.to(weights_at.scalar_type()));
    indices_at.copy_(selected.to(indices_at.scalar_type()));
#else
    (void)topk_weights;
    (void)topk_indices;
    (void)router_logits;
    (void)input_ids;
    (void)tid2eid;
    (void)renormalize;
    throw std::runtime_error("deepseek_v4_hash_topk_naive_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}


void deepseek_v4_hash_topk_kernel_(Tensor topk_weights,
                                   Tensor topk_indices,
                                   const Tensor &router_logits,
                                   const Tensor &input_ids,
                                   const Tensor &tid2eid,
                                   bool renormalize) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(router_logits, "deepseek_v4_hash_topk_kernel_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    check_shapes(topk_weights, topk_indices, router_logits, input_ids, tid2eid);
    check_kernel_tensors(topk_weights, topk_indices, router_logits, input_ids, tid2eid);
    if (tid2eid->size(1) > 32) {
        throw std::runtime_error("deepseek_v4_hash_topk_kernel_ supports topk <= 32.");
    }

    deepseek_v4_hash_topk::launch_hash_topk(
        reinterpret_cast<float *>(topk_weights->data()),
        reinterpret_cast<int32_t *>(topk_indices->data()),
        reinterpret_cast<const float *>(router_logits->data()),
        reinterpret_cast<const int64_t *>(input_ids->data()),
        tid2eid->data(),
        tid2eid->dtype() == DataType::I64,
        static_cast<int64_t>(router_logits->size(0)),
        static_cast<int64_t>(router_logits->size(1)),
        static_cast<int64_t>(tid2eid->size(1)),
        renormalize,
        current_accelerator_stream());
#else
    (void)topk_weights;
    (void)topk_indices;
    (void)router_logits;
    (void)input_ids;
    (void)tid2eid;
    (void)renormalize;
    throw std::runtime_error("deepseek_v4_hash_topk_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}


void deepseek_v4_hash_topk_generic_kernel_(Tensor topk_weights,
                                           Tensor topk_indices,
                                           const Tensor &router_logits,
                                           const Tensor &input_ids,
                                           const Tensor &tid2eid,
                                           bool renormalize) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(router_logits, "deepseek_v4_hash_topk_generic_kernel_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    check_shapes(topk_weights, topk_indices, router_logits, input_ids, tid2eid);
    check_kernel_tensors(topk_weights, topk_indices, router_logits, input_ids, tid2eid);
    if (tid2eid->size(1) > 32) {
        throw std::runtime_error("deepseek_v4_hash_topk_generic_kernel_ supports topk <= 32.");
    }

    deepseek_v4_hash_topk::launch_hash_topk_generic(
        reinterpret_cast<float *>(topk_weights->data()),
        reinterpret_cast<int32_t *>(topk_indices->data()),
        reinterpret_cast<const float *>(router_logits->data()),
        reinterpret_cast<const int64_t *>(input_ids->data()),
        tid2eid->data(),
        tid2eid->dtype() == DataType::I64,
        static_cast<int64_t>(router_logits->size(0)),
        static_cast<int64_t>(router_logits->size(1)),
        static_cast<int64_t>(tid2eid->size(1)),
        renormalize,
        current_accelerator_stream());
#else
    (void)topk_weights;
    (void)topk_indices;
    (void)router_logits;
    (void)input_ids;
    (void)tid2eid;
    (void)renormalize;
    throw std::runtime_error("deepseek_v4_hash_topk_generic_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_hash_topk_dsv4_kernel_(Tensor topk_weights,
                                        Tensor topk_indices,
                                        const Tensor &router_logits,
                                        const Tensor &input_ids,
                                        const Tensor &tid2eid,
                                        bool renormalize) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_accelerator_tensor(router_logits, "deepseek_v4_hash_topk_dsv4_kernel_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    check_shapes(topk_weights, topk_indices, router_logits, input_ids, tid2eid);
    check_kernel_tensors(topk_weights, topk_indices, router_logits, input_ids, tid2eid);
    if (router_logits->size(1) != 256 || tid2eid->size(1) != 6 || topk_weights->size(1) != 6 || !renormalize) {
        throw std::runtime_error("deepseek_v4_hash_topk_dsv4_kernel_ requires num_experts=256, topk=6, renormalize=true.");
    }

    deepseek_v4_hash_topk::launch_hash_topk_dsv4(
        reinterpret_cast<float *>(topk_weights->data()),
        reinterpret_cast<int32_t *>(topk_indices->data()),
        reinterpret_cast<const float *>(router_logits->data()),
        reinterpret_cast<const int64_t *>(input_ids->data()),
        tid2eid->data(),
        tid2eid->dtype() == DataType::I64,
        static_cast<int64_t>(router_logits->size(0)),
        current_accelerator_stream());
#else
    (void)topk_weights;
    (void)topk_indices;
    (void)router_logits;
    (void)input_ids;
    (void)tid2eid;
    (void)renormalize;
    throw std::runtime_error("deepseek_v4_hash_topk_dsv4_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
