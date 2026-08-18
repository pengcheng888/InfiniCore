#include "infinicore/ops/deepseek_v4_biased_topk.hpp"

#include "deepseek_v4_biased_topk_kernel.hpp"

#include "../../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/graph/graph.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4TopkDsv4Kernel, Tensor, Tensor, const Tensor &, const Tensor &, bool);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4TopkDsv4Kernel);

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
                  const Tensor &correction_bias) {
    if (router_logits->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_topk_kernel_ expects router_logits to be 2-D.");
    }
    if (correction_bias->ndim() != 1 || correction_bias->size(0) != router_logits->size(1)) {
        throw std::runtime_error("deepseek_v4_topk_kernel_ correction_bias shape mismatch.");
    }
    if (topk_weights->shape() != topk_indices->shape()) {
        throw std::runtime_error("deepseek_v4_topk_kernel_ topk weight/index shape mismatch.");
    }
    if (topk_weights->ndim() != 2 || topk_weights->size(0) != router_logits->size(0)) {
        throw std::runtime_error("deepseek_v4_topk_kernel_ output shape mismatch.");
    }
}


#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
void check_kernel_tensors(const Tensor &topk_weights,
                          const Tensor &topk_indices,
                          const Tensor &router_logits,
                          const Tensor &correction_bias) {
    if (topk_weights->dtype() != DataType::F32 || topk_indices->dtype() != DataType::I32 ||
        router_logits->dtype() != DataType::F32 || correction_bias->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_topk_kernel_ expects F32 weights/logits/bias and I32 indices.");
    }
    if (!topk_weights->is_contiguous() || !topk_indices->is_contiguous() ||
        !router_logits->is_contiguous() || !correction_bias->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_topk_kernel_ expects contiguous tensors.");
    }
}
#endif

bool is_dsv4_topk_shape(const Tensor &topk_weights,
                        const Tensor &router_logits,
                        bool renormalize) {
    return router_logits->size(1) == 256 && topk_weights->size(1) == 6 && renormalize;
}

} // namespace

DeepseekV4TopkDsv4Kernel::DeepseekV4TopkDsv4Kernel(Tensor topk_weights,
                                                   Tensor topk_indices,
                                                   const Tensor &router_logits,
                                                   const Tensor &correction_bias,
                                                   bool renormalize) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_indices, router_logits, correction_bias);
    INFINICORE_GRAPH_OP_DISPATCH(topk_weights->device().getType(), topk_weights, topk_indices, router_logits, correction_bias, renormalize);
}

void DeepseekV4TopkDsv4Kernel::execute(Tensor topk_weights,
                                       Tensor topk_indices,
                                       const Tensor &router_logits,
                                       const Tensor &correction_bias,
                                       bool renormalize) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4TopkDsv4Kernel, topk_weights, topk_indices, router_logits, correction_bias, renormalize);
}

namespace deepseek_v4_biased_topk_graph_impl {

struct Dsv4PlannedMeta {
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_indices;
    graph::GraphTensor router_logits;
    graph::GraphTensor correction_bias;
    int64_t tokens;
};

void *plan_dsv4(Tensor topk_weights,
                Tensor topk_indices,
                const Tensor &router_logits,
                const Tensor &correction_bias,
                bool renormalize) {
    check_accelerator_tensor(router_logits, "DeepseekV4TopkDsv4Kernel");
    check_shapes(topk_weights, topk_indices, router_logits, correction_bias);
    check_kernel_tensors(topk_weights, topk_indices, router_logits, correction_bias);
    if (!is_dsv4_topk_shape(topk_weights, router_logits, renormalize)) {
        throw std::runtime_error("DeepseekV4TopkDsv4Kernel requires num_experts=256, topk=6, renormalize=true.");
    }
    return new Dsv4PlannedMeta{
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_indices),
        graph::GraphTensor(router_logits),
        graph::GraphTensor(correction_bias),
        static_cast<int64_t>(router_logits->size(0))};
}

void run_dsv4(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<Dsv4PlannedMeta *>(planned_meta);
    deepseek_v4_biased_topk::launch_biased_topk_dsv4(
        reinterpret_cast<float *>(planned->topk_weights->data()),
        reinterpret_cast<int32_t *>(planned->topk_indices->data()),
        reinterpret_cast<const float *>(planned->router_logits->data()),
        reinterpret_cast<const float *>(planned->correction_bias->data()),
        planned->tokens,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("DeepseekV4TopkDsv4Kernel requires a HYGON/NVIDIA build.");
#endif
}

void cleanup_dsv4(void **planned_meta_ptr) {
    delete *reinterpret_cast<Dsv4PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_biased_topk_graph_impl

namespace deepseek_v4_biased_topk_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4TopkDsv4Kernel,
                                       &deepseek_v4_biased_topk_graph_impl::plan_dsv4,
                                       &deepseek_v4_biased_topk_graph_impl::run_dsv4,
                                       &deepseek_v4_biased_topk_graph_impl::cleanup_dsv4);
} // namespace deepseek_v4_biased_topk_register

namespace {

void run_biased_topk_dsv4_kernel(Tensor topk_weights,
                                 Tensor topk_indices,
                                 const Tensor &router_logits,
                                 const Tensor &correction_bias,
                                 bool renormalize) {
    DeepseekV4TopkDsv4Kernel::execute(topk_weights, topk_indices, router_logits, correction_bias, renormalize);
}

} // namespace

void deepseek_v4_topk_kernel_(Tensor topk_weights,
                              Tensor topk_indices,
                              const Tensor &router_logits,
                              const Tensor &correction_bias,
                              bool renormalize) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    check_accelerator_tensor(router_logits, "deepseek_v4_topk_kernel_");

    check_shapes(topk_weights, topk_indices, router_logits, correction_bias);
    check_kernel_tensors(topk_weights, topk_indices, router_logits, correction_bias);
    if (router_logits->size(1) > 512 || topk_weights->size(1) > 16) {
        throw std::runtime_error("deepseek_v4_topk_kernel_ supports num_experts <= 512 and topk <= 16.");
    }
    if (is_dsv4_topk_shape(topk_weights, router_logits, renormalize)) {
        // DeepSeek V4 fixed-shape fast path. Keep this specialization internal;
        // callers should use deepseek_v4_topk_kernel_.
        run_biased_topk_dsv4_kernel(topk_weights, topk_indices, router_logits, correction_bias, renormalize);
        return;
    }

    deepseek_v4_biased_topk::launch_biased_topk(
        reinterpret_cast<float *>(topk_weights->data()),
        reinterpret_cast<int32_t *>(topk_indices->data()),
        reinterpret_cast<const float *>(router_logits->data()),
        reinterpret_cast<const float *>(correction_bias->data()),
        static_cast<int64_t>(router_logits->size(0)),
        static_cast<int64_t>(router_logits->size(1)),
        static_cast<int64_t>(topk_weights->size(1)),
        renormalize,
        context::getStream());
#else
    (void)topk_weights;
    (void)topk_indices;
    (void)router_logits;
    (void)correction_bias;
    (void)renormalize;
    throw std::runtime_error("deepseek_v4_topk_kernel_ requires a HYGON/NVIDIA build.");
#endif
}


void deepseek_v4_topk_generic_kernel_(Tensor topk_weights,
                                      Tensor topk_indices,
                                      const Tensor &router_logits,
                                      const Tensor &correction_bias,
                                      bool renormalize) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    check_accelerator_tensor(router_logits, "deepseek_v4_topk_generic_kernel_");

    check_shapes(topk_weights, topk_indices, router_logits, correction_bias);
    check_kernel_tensors(topk_weights, topk_indices, router_logits, correction_bias);
    if (router_logits->size(1) > 512 || topk_weights->size(1) > 16) {
        throw std::runtime_error("deepseek_v4_topk_generic_kernel_ supports num_experts <= 512 and topk <= 16.");
    }

    deepseek_v4_biased_topk::launch_biased_topk_generic(
        reinterpret_cast<float *>(topk_weights->data()),
        reinterpret_cast<int32_t *>(topk_indices->data()),
        reinterpret_cast<const float *>(router_logits->data()),
        reinterpret_cast<const float *>(correction_bias->data()),
        static_cast<int64_t>(router_logits->size(0)),
        static_cast<int64_t>(router_logits->size(1)),
        static_cast<int64_t>(topk_weights->size(1)),
        renormalize,
        context::getStream());
#else
    (void)topk_weights;
    (void)topk_indices;
    (void)router_logits;
    (void)correction_bias;
    (void)renormalize;
    throw std::runtime_error("deepseek_v4_topk_generic_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
