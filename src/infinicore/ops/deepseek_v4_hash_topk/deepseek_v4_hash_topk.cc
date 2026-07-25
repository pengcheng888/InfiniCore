#include "infinicore/ops/deepseek_v4_hash_topk.hpp"

#include "deepseek_v4_hash_topk_kernel.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/graph/graph.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4HashTopkDsv4Kernel, Tensor, Tensor, const Tensor &, const Tensor &, const Tensor &, bool);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4HashTopkDsv4Kernel);

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


#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
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
#endif

bool is_dsv4_hash_topk_shape(const Tensor &topk_weights,
                             const Tensor &router_logits,
                             const Tensor &tid2eid,
                             bool renormalize) {
    return router_logits->size(1) == 256 && tid2eid->size(1) == 6 && topk_weights->size(1) == 6 && renormalize;
}

} // namespace

DeepseekV4HashTopkDsv4Kernel::DeepseekV4HashTopkDsv4Kernel(Tensor topk_weights,
                                                           Tensor topk_indices,
                                                           const Tensor &router_logits,
                                                           const Tensor &input_ids,
                                                           const Tensor &tid2eid,
                                                           bool renormalize) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_indices, router_logits, input_ids, tid2eid);
    INFINICORE_GRAPH_OP_DISPATCH(topk_weights->device().getType(), topk_weights, topk_indices, router_logits, input_ids, tid2eid, renormalize);
}

void DeepseekV4HashTopkDsv4Kernel::execute(Tensor topk_weights,
                                           Tensor topk_indices,
                                           const Tensor &router_logits,
                                           const Tensor &input_ids,
                                           const Tensor &tid2eid,
                                           bool renormalize) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4HashTopkDsv4Kernel, topk_weights, topk_indices, router_logits, input_ids, tid2eid, renormalize);
}

namespace deepseek_v4_hash_topk_graph_impl {

struct Dsv4PlannedMeta {
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_indices;
    graph::GraphTensor router_logits;
    graph::GraphTensor input_ids;
    graph::GraphTensor tid2eid;
    bool tid2eid_i64;
    int64_t tokens;
};

void *plan_dsv4(Tensor topk_weights,
                Tensor topk_indices,
                const Tensor &router_logits,
                const Tensor &input_ids,
                const Tensor &tid2eid,
                bool renormalize) {
    check_accelerator_tensor(router_logits, "DeepseekV4HashTopkDsv4Kernel");
    check_shapes(topk_weights, topk_indices, router_logits, input_ids, tid2eid);
    check_kernel_tensors(topk_weights, topk_indices, router_logits, input_ids, tid2eid);
    if (!is_dsv4_hash_topk_shape(topk_weights, router_logits, tid2eid, renormalize)) {
        throw std::runtime_error("DeepseekV4HashTopkDsv4Kernel requires num_experts=256, topk=6, renormalize=true.");
    }
    return new Dsv4PlannedMeta{
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_indices),
        graph::GraphTensor(router_logits),
        graph::GraphTensor(input_ids),
        graph::GraphTensor(tid2eid),
        tid2eid->dtype() == DataType::I64,
        static_cast<int64_t>(router_logits->size(0))};
}

void run_dsv4(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<Dsv4PlannedMeta *>(planned_meta);
    deepseek_v4_hash_topk::launch_hash_topk_dsv4(
        reinterpret_cast<float *>(planned->topk_weights->data()),
        reinterpret_cast<int32_t *>(planned->topk_indices->data()),
        reinterpret_cast<const float *>(planned->router_logits->data()),
        reinterpret_cast<const int64_t *>(planned->input_ids->data()),
        planned->tid2eid->data(),
        planned->tid2eid_i64,
        planned->tokens,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("DeepseekV4HashTopkDsv4Kernel requires a HYGON/NVIDIA build.");
#endif
}

void cleanup_dsv4(void **planned_meta_ptr) {
    delete *reinterpret_cast<Dsv4PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_hash_topk_graph_impl

namespace deepseek_v4_hash_topk_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4HashTopkDsv4Kernel,
                                       &deepseek_v4_hash_topk_graph_impl::plan_dsv4,
                                       &deepseek_v4_hash_topk_graph_impl::run_dsv4,
                                       &deepseek_v4_hash_topk_graph_impl::cleanup_dsv4);
} // namespace deepseek_v4_hash_topk_register

namespace {

void run_hash_topk_dsv4_kernel(Tensor topk_weights,
                               Tensor topk_indices,
                               const Tensor &router_logits,
                               const Tensor &input_ids,
                               const Tensor &tid2eid,
                               bool renormalize) {
    DeepseekV4HashTopkDsv4Kernel::execute(topk_weights, topk_indices, router_logits, input_ids, tid2eid, renormalize);
}

} // namespace

void deepseek_v4_hash_topk_kernel_(Tensor topk_weights,
                                   Tensor topk_indices,
                                   const Tensor &router_logits,
                                   const Tensor &input_ids,
                                   const Tensor &tid2eid,
                                   bool renormalize) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    check_accelerator_tensor(router_logits, "deepseek_v4_hash_topk_kernel_");

    check_shapes(topk_weights, topk_indices, router_logits, input_ids, tid2eid);
    check_kernel_tensors(topk_weights, topk_indices, router_logits, input_ids, tid2eid);
    if (tid2eid->size(1) > 32) {
        throw std::runtime_error("deepseek_v4_hash_topk_kernel_ supports topk <= 32.");
    }
    if (is_dsv4_hash_topk_shape(topk_weights, router_logits, tid2eid, renormalize)) {
        // DeepSeek V4 fixed-shape fast path. Keep this specialization internal;
        // callers should use deepseek_v4_hash_topk_kernel_.
        run_hash_topk_dsv4_kernel(topk_weights, topk_indices, router_logits, input_ids, tid2eid, renormalize);
        return;
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
        context::getStream());
#else
    (void)topk_weights;
    (void)topk_indices;
    (void)router_logits;
    (void)input_ids;
    (void)tid2eid;
    (void)renormalize;
    throw std::runtime_error("deepseek_v4_hash_topk_kernel_ requires a HYGON/NVIDIA build.");
#endif
}


void deepseek_v4_hash_topk_generic_kernel_(Tensor topk_weights,
                                           Tensor topk_indices,
                                           const Tensor &router_logits,
                                           const Tensor &input_ids,
                                           const Tensor &tid2eid,
                                           bool renormalize) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    check_accelerator_tensor(router_logits, "deepseek_v4_hash_topk_generic_kernel_");

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
        context::getStream());
#else
    (void)topk_weights;
    (void)topk_indices;
    (void)router_logits;
    (void)input_ids;
    (void)tid2eid;
    (void)renormalize;
    throw std::runtime_error("deepseek_v4_hash_topk_generic_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
