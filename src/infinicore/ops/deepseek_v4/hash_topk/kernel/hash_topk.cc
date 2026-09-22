#include "infinicore/ops/deepseek_v4/hash_topk.hpp"

#include "hash_topk_kernel.hpp"

#include "../../platform.hpp"
#include "../../../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/graph/graph.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op::deepseek_v4 {

INFINICORE_GRAPH_OP_CLASS(HashTopkKernel, Tensor, Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, float);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(HashTopkKernel);

namespace {

void check_accelerator_tensor(const Tensor &tensor, const char *op_name) {
    detail::check_build_device(tensor, op_name);
}

void check_shapes(const Tensor &topk_weights,
                  const Tensor &topk_indices,
                  const Tensor &router_logits,
                  const Tensor &input_ids,
                  const Tensor &tid2eid,
                  int64_t num_fused_shared_experts) {
    if (router_logits->ndim() != 2) {
        throw std::runtime_error("hash_topk_ expects router_logits to be 2-D.");
    }
    if (input_ids->ndim() != 1) {
        throw std::runtime_error("hash_topk_ expects input_ids to be 1-D.");
    }
    if (tid2eid->ndim() != 2) {
        throw std::runtime_error("hash_topk_ expects tid2eid to be 2-D.");
    }
    if (topk_weights->shape() != topk_indices->shape()) {
        throw std::runtime_error("hash_topk_ topk weight/index shape mismatch.");
    }
    if (num_fused_shared_experts < 0) {
        throw std::runtime_error("hash_topk_ expects num_fused_shared_experts >= 0.");
    }
    if (topk_weights->shape() != Shape{router_logits->size(0), tid2eid->size(1) + num_fused_shared_experts}) {
        throw std::runtime_error("hash_topk_ output shape mismatch.");
    }
    if (input_ids->size(0) != router_logits->size(0)) {
        throw std::runtime_error("hash_topk_ input_ids/router token count mismatch.");
    }
}

void check_scoring_config(float routed_scaling_factor, const std::string &scoring_func) {
    if (scoring_func != "sqrtsoftplus") {
        throw std::runtime_error("hash_topk_ only supports scoring_func='sqrtsoftplus'.");
    }
    if (routed_scaling_factor == 0.0f) {
        throw std::runtime_error("hash_topk_ expects routed_scaling_factor != 0.");
    }
}

void check_kernel_tensors(const Tensor &topk_weights,
                          const Tensor &topk_indices,
                          const Tensor &router_logits,
                          const Tensor &input_ids,
                          const Tensor &tid2eid) {
    if (topk_weights->dtype() != DataType::F32 || topk_indices->dtype() != DataType::I32 || router_logits->dtype() != DataType::F32 || input_ids->dtype() != DataType::I64 || (tid2eid->dtype() != DataType::I64 && tid2eid->dtype() != DataType::I32)) {
        throw std::runtime_error("hash_topk_kernel_ expects F32 weights/logits, I32 indices, I64 input_ids, and I64/I32 tid2eid.");
    }
    if (!topk_weights->is_contiguous() || !topk_indices->is_contiguous() || !router_logits->is_contiguous() || !input_ids->is_contiguous() || !tid2eid->is_contiguous()) {
        throw std::runtime_error("hash_topk_kernel_ expects contiguous tensors.");
    }
}

} // namespace

HashTopkKernel::HashTopkKernel(Tensor topk_weights,
                                                   Tensor topk_indices,
                                                   const Tensor &router_logits,
                                                   const Tensor &input_ids,
                                                   const Tensor &tid2eid,
                                                   int64_t num_fused_shared_experts,
                                                   float routed_scaling_factor) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_indices, router_logits, input_ids, tid2eid);
    INFINICORE_GRAPH_OP_DISPATCH(topk_weights->device().getType(), topk_weights, topk_indices, router_logits, input_ids, tid2eid, num_fused_shared_experts, routed_scaling_factor);
}

void HashTopkKernel::execute(Tensor topk_weights,
                                       Tensor topk_indices,
                                       const Tensor &router_logits,
                                       const Tensor &input_ids,
                                       const Tensor &tid2eid,
                                       int64_t num_fused_shared_experts,
                                       float routed_scaling_factor) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(HashTopkKernel, topk_weights, topk_indices, router_logits, input_ids, tid2eid, num_fused_shared_experts, routed_scaling_factor);
}

namespace hash_topk_graph_impl {

struct HashTopkPlannedMeta {
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_indices;
    graph::GraphTensor router_logits;
    graph::GraphTensor input_ids;
    graph::GraphTensor tid2eid;
    bool tid2eid_i64;
    int64_t tokens;
    int64_t num_experts;
    int64_t topk;
    int64_t num_fused_shared_experts;
    float routed_scaling_factor;
};

void *plan_hash_topk_common(Tensor topk_weights,
                            Tensor topk_indices,
                            const Tensor &router_logits,
                            const Tensor &input_ids,
                            const Tensor &tid2eid,
                            int64_t num_fused_shared_experts,
                            float routed_scaling_factor,
                            int64_t fused_topk_limit,
                            const char *op_name) {
    check_accelerator_tensor(router_logits, op_name);
    check_scoring_config(routed_scaling_factor, "sqrtsoftplus");
    check_shapes(topk_weights, topk_indices, router_logits, input_ids, tid2eid, num_fused_shared_experts);
    check_kernel_tensors(topk_weights, topk_indices, router_logits, input_ids, tid2eid);
    if (tid2eid->size(1) + num_fused_shared_experts > fused_topk_limit) {
        throw std::runtime_error(std::string(op_name) + " supports fused topk <= backend limit.");
    }
    return new HashTopkPlannedMeta{
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_indices),
        graph::GraphTensor(router_logits),
        graph::GraphTensor(input_ids),
        graph::GraphTensor(tid2eid),
        tid2eid->dtype() == DataType::I64,
        static_cast<int64_t>(router_logits->size(0)),
        static_cast<int64_t>(router_logits->size(1)),
        static_cast<int64_t>(tid2eid->size(1)),
        num_fused_shared_experts,
        routed_scaling_factor};
}

void *plan_hash_topk(Tensor topk_weights,
                     Tensor topk_indices,
                     const Tensor &router_logits,
                     const Tensor &input_ids,
                     const Tensor &tid2eid,
                     int64_t num_fused_shared_experts,
                     float routed_scaling_factor) {
    return plan_hash_topk_common(topk_weights,
                                 topk_indices,
                                 router_logits,
                                 input_ids,
                                 tid2eid,
                                 num_fused_shared_experts,
                                 routed_scaling_factor,
                                 32,
                                 "HashTopkKernel");
}

void run_hash_topk(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    auto *planned = reinterpret_cast<HashTopkPlannedMeta *>(planned_meta);
    auto *topk_weights = reinterpret_cast<float *>(planned->topk_weights->data());
    auto *topk_indices = reinterpret_cast<int32_t *>(planned->topk_indices->data());
    const auto *router_logits = reinterpret_cast<const float *>(planned->router_logits->data());
    const auto *input_ids = reinterpret_cast<const int64_t *>(planned->input_ids->data());
    void *tid2eid = planned->tid2eid->data();
    hash_topk::launch_hash_topk(
        topk_weights,
        topk_indices,
        router_logits,
        input_ids,
        tid2eid,
        planned->tid2eid_i64,
        planned->tokens,
        planned->num_experts,
        planned->topk,
        planned->num_fused_shared_experts,
        planned->routed_scaling_factor,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("HashTopkKernel requires a HYGON/NVIDIA/METAX build.");
#endif
}

void cleanup_hash_topk(void **planned_meta_ptr) {
    delete *reinterpret_cast<HashTopkPlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace hash_topk_graph_impl

namespace hash_topk_register {
INFINICORE_DSV4_NATIVE_GRAPH_OP_REGISTER_BUILD_DEVICE(
    HashTopkKernel,
    &hash_topk_graph_impl::plan_hash_topk,
    &hash_topk_graph_impl::run_hash_topk,
    &hash_topk_graph_impl::cleanup_hash_topk);
} // namespace hash_topk_register

void hash_topk_kernel_(Tensor topk_weights,
                                   Tensor topk_indices,
                                   const Tensor &router_logits,
                                   const Tensor &input_ids,
                                   const Tensor &tid2eid,
                                   int64_t num_fused_shared_experts,
                                   float routed_scaling_factor,
                                   const std::string &scoring_func) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    check_scoring_config(routed_scaling_factor, scoring_func);
    HashTopkKernel::execute(topk_weights,
                                      topk_indices,
                                      router_logits,
                                      input_ids,
                                      tid2eid,
                                      num_fused_shared_experts,
                                      routed_scaling_factor);
#else
    (void)topk_weights;
    (void)topk_indices;
    (void)router_logits;
    (void)input_ids;
    (void)tid2eid;
    (void)num_fused_shared_experts;
    (void)routed_scaling_factor;
    (void)scoring_func;
    throw std::runtime_error("hash_topk_kernel_ requires a HYGON/NVIDIA/METAX build.");
#endif
}

void hash_topk_generic_kernel_(Tensor topk_weights,
                                           Tensor topk_indices,
                                           const Tensor &router_logits,
                                           const Tensor &input_ids,
                                           const Tensor &tid2eid,
                                           int64_t num_fused_shared_experts,
                                           float routed_scaling_factor,
                                           const std::string &scoring_func) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    check_accelerator_tensor(router_logits, "hash_topk_generic_kernel_");

    check_scoring_config(routed_scaling_factor, scoring_func);
    check_shapes(topk_weights, topk_indices, router_logits, input_ids, tid2eid, num_fused_shared_experts);
    check_kernel_tensors(topk_weights, topk_indices, router_logits, input_ids, tid2eid);
    if (tid2eid->size(1) + num_fused_shared_experts > 32) {
        throw std::runtime_error("hash_topk_generic_kernel_ supports fused topk <= 32.");
    }

    hash_topk::launch_hash_topk_generic(
        reinterpret_cast<float *>(topk_weights->data()),
        reinterpret_cast<int32_t *>(topk_indices->data()),
        reinterpret_cast<const float *>(router_logits->data()),
        reinterpret_cast<const int64_t *>(input_ids->data()),
        tid2eid->data(),
        tid2eid->dtype() == DataType::I64,
        static_cast<int64_t>(router_logits->size(0)),
        static_cast<int64_t>(router_logits->size(1)),
        static_cast<int64_t>(tid2eid->size(1)),
        num_fused_shared_experts,
        routed_scaling_factor,
        context::getStream());
#else
    (void)topk_weights;
    (void)topk_indices;
    (void)router_logits;
    (void)input_ids;
    (void)tid2eid;
    (void)num_fused_shared_experts;
    (void)routed_scaling_factor;
    (void)scoring_func;
    throw std::runtime_error("hash_topk_generic_kernel_ requires a HYGON/NVIDIA/METAX build.");
#endif
}

} // namespace infinicore::op::deepseek_v4
