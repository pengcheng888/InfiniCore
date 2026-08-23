#include "infinicore/ops/deepseek_v4_fused_q_indexer_rope_hadamard_quant.hpp"

#include "deepseek_v4_fused_q_indexer_rope_hadamard_quant_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4FusedQIndexerRopeHadamardQuantKernel);

namespace {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kRopeDim = 64;

void check_accel_tensor(const Tensor &tensor, const char *op_name) {
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

int scalar_type_for_kernel(const Tensor &tensor, const char *op_name) {
    if (tensor->dtype() == DataType::BF16) {
        return deepseek_v4_fused_q_indexer_rope_hadamard_quant::kDsv4BF16;
    }
    if (tensor->dtype() == DataType::F16) {
        return deepseek_v4_fused_q_indexer_rope_hadamard_quant::kDsv4F16;
    }
    if (tensor->dtype() == DataType::F32) {
        return deepseek_v4_fused_q_indexer_rope_hadamard_quant::kDsv4F32;
    }
    throw std::runtime_error(std::string(op_name) + " supports bf16/fp16/fp32 q and weights.");
}

void check_shapes(const Tensor &q,
                  const Tensor &indexer_weights,
                  const Tensor &q_fp8,
                  const Tensor &q_scale,
                  const Tensor &fused_weights,
                  const Tensor &freqs_cis,
                  const Tensor &positions,
                  const char *op_name) {
    check_accel_tensor(q, op_name);
    check_accel_tensor(indexer_weights, op_name);
    check_accel_tensor(q_fp8, op_name);
    check_accel_tensor(q_scale, op_name);
    check_accel_tensor(fused_weights, op_name);
    check_accel_tensor(freqs_cis, op_name);
    check_accel_tensor(positions, op_name);

    if (q->ndim() != 3 || q->size(2) != static_cast<size_t>(kHeadDim)) {
        throw std::runtime_error(std::string(op_name) + " expects q [tokens, heads, 128].");
    }
    scalar_type_for_kernel(q, op_name);
    if (indexer_weights->ndim() != 2 || indexer_weights->size(0) != q->size(0) || indexer_weights->size(1) != q->size(1)) {
        throw std::runtime_error(std::string(op_name) + " expects indexer_weights [tokens, heads].");
    }
    scalar_type_for_kernel(indexer_weights, op_name);
    if (q_fp8->shape() != q->shape() || q_fp8->dtype() != DataType::F8) {
        throw std::runtime_error(std::string(op_name) + " expects q_fp8 [tokens, heads, 128] f8.");
    }
    if (q_scale->ndim() != 3 || q_scale->size(0) != q->size(0) || q_scale->size(1) != q->size(1) || q_scale->size(2) != 1 || q_scale->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " expects q_scale [tokens, heads, 1] fp32.");
    }
    if (fused_weights->shape() != indexer_weights->shape() || fused_weights->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " expects fused_weights [tokens, heads] fp32.");
    }
    if (freqs_cis->ndim() != 2 || freqs_cis->size(1) != static_cast<size_t>(kRopeDim) || freqs_cis->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " expects freqs_cis [max_pos, 64] fp32.");
    }
    if (positions->ndim() != 1 || positions->size(0) != q->size(0) ||
        (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64)) {
        throw std::runtime_error(std::string(op_name) + " expects positions [tokens] int32/int64.");
    }
    if (!q->is_contiguous() || !indexer_weights->is_contiguous() || !q_fp8->is_contiguous() ||
        !q_scale->is_contiguous() || !fused_weights->is_contiguous() || !freqs_cis->is_contiguous() ||
        !positions->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
}

void fused_q_indexer_rope_hadamard_quant_impl(const Tensor &q,
                                              const Tensor &indexer_weights,
                                              Tensor q_fp8,
                                              Tensor q_scale,
                                              Tensor fused_weights,
                                              float weight_scale,
                                              const Tensor &freqs_cis,
                                              const Tensor &positions) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    constexpr const char *op_name = "deepseek_v4_fused_q_indexer_rope_hadamard_quant_kernel_";
    check_shapes(q, indexer_weights, q_fp8, q_scale, fused_weights, freqs_cis, positions, op_name);
    deepseek_v4_fused_q_indexer_rope_hadamard_quant::launch_fused_q_indexer_rope_hadamard_quant(
        q->data(),
        scalar_type_for_kernel(q, op_name),
        indexer_weights->data(),
        scalar_type_for_kernel(indexer_weights, op_name),
        reinterpret_cast<uint8_t *>(q_fp8->data()),
        reinterpret_cast<float *>(q_scale->data()),
        reinterpret_cast<float *>(fused_weights->data()),
        weight_scale,
        reinterpret_cast<const float *>(freqs_cis->data()),
        positions->data(),
        positions->dtype() == DataType::I64,
        static_cast<int64_t>(q->size(0) * q->size(1)),
        static_cast<int64_t>(q->size(1)),
        context::getStream());
#else
    (void)q;
    (void)indexer_weights;
    (void)q_fp8;
    (void)q_scale;
    (void)fused_weights;
    (void)weight_scale;
    (void)freqs_cis;
    (void)positions;
    throw std::runtime_error("deepseek_v4_fused_q_indexer_rope_hadamard_quant_kernel_ requires a HYGON/NVIDIA build.");
#endif
    return;
}

} // namespace

DeepseekV4FusedQIndexerRopeHadamardQuantKernel::DeepseekV4FusedQIndexerRopeHadamardQuantKernel(
    const Tensor &q,
    const Tensor &indexer_weights,
    Tensor q_fp8,
    Tensor q_scale,
    Tensor fused_weights,
    float weight_scale,
    const Tensor &freqs_cis,
    const Tensor &positions) {
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(), q, indexer_weights, q_fp8, q_scale, fused_weights, weight_scale, freqs_cis, positions);
}

void DeepseekV4FusedQIndexerRopeHadamardQuantKernel::execute(const Tensor &q,
                                                             const Tensor &indexer_weights,
                                                             Tensor q_fp8,
                                                             Tensor q_scale,
                                                             Tensor fused_weights,
                                                             float weight_scale,
                                                             const Tensor &freqs_cis,
                                                             const Tensor &positions) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4FusedQIndexerRopeHadamardQuantKernel, q, indexer_weights, q_fp8, q_scale, fused_weights, weight_scale, freqs_cis, positions);
    return;
}

namespace deepseek_v4_fused_q_indexer_rope_hadamard_quant_graph_impl {

struct PlannedMeta {
    graph::GraphTensor q;
    graph::GraphTensor indexer_weights;
    graph::GraphTensor q_fp8;
    graph::GraphTensor q_scale;
    graph::GraphTensor fused_weights;
    graph::GraphTensor freqs_cis;
    graph::GraphTensor positions;
    float weight_scale;
};

void *plan(const Tensor &q,
           const Tensor &indexer_weights,
           Tensor q_fp8,
           Tensor q_scale,
           Tensor fused_weights,
           float weight_scale,
           const Tensor &freqs_cis,
           const Tensor &positions) {
    return new PlannedMeta{graph::GraphTensor(q),
                           graph::GraphTensor(indexer_weights),
                           graph::GraphTensor(q_fp8),
                           graph::GraphTensor(q_scale),
                           graph::GraphTensor(fused_weights),
                           graph::GraphTensor(freqs_cis),
                           graph::GraphTensor(positions),
                           weight_scale};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    fused_q_indexer_rope_hadamard_quant_impl(planned->q,
                                             planned->indexer_weights,
                                             planned->q_fp8,
                                             planned->q_scale,
                                             planned->fused_weights,
                                             planned->weight_scale,
                                             planned->freqs_cis,
                                             planned->positions);
    return;
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
    return;
}

} // namespace deepseek_v4_fused_q_indexer_rope_hadamard_quant_graph_impl

namespace deepseek_v4_fused_q_indexer_rope_hadamard_quant_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4FusedQIndexerRopeHadamardQuantKernel,
                                       &deepseek_v4_fused_q_indexer_rope_hadamard_quant_graph_impl::plan,
                                       &deepseek_v4_fused_q_indexer_rope_hadamard_quant_graph_impl::run,
                                       &deepseek_v4_fused_q_indexer_rope_hadamard_quant_graph_impl::cleanup);
} // namespace deepseek_v4_fused_q_indexer_rope_hadamard_quant_register

void deepseek_v4_fused_q_indexer_rope_hadamard_quant_(const Tensor &q,
                                                      const Tensor &indexer_weights,
                                                      Tensor q_fp8,
                                                      Tensor q_scale,
                                                      Tensor fused_weights,
                                                      float weight_scale,
                                                      const Tensor &freqs_cis,
                                                      const Tensor &positions) {
    deepseek_v4_fused_q_indexer_rope_hadamard_quant_kernel_(q,
                                                            indexer_weights,
                                                            q_fp8,
                                                            q_scale,
                                                            fused_weights,
                                                            weight_scale,
                                                            freqs_cis,
                                                            positions);
    return;
}

void deepseek_v4_fused_q_indexer_rope_hadamard_quant_kernel_(const Tensor &q,
                                                             const Tensor &indexer_weights,
                                                             Tensor q_fp8,
                                                             Tensor q_scale,
                                                             Tensor fused_weights,
                                                             float weight_scale,
                                                             const Tensor &freqs_cis,
                                                             const Tensor &positions) {
    DeepseekV4FusedQIndexerRopeHadamardQuantKernel::execute(q, indexer_weights, q_fp8, q_scale, fused_weights, weight_scale, freqs_cis, positions);
    return;
}

} // namespace infinicore::op
