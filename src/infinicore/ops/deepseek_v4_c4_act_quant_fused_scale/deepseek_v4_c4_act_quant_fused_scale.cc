#include "infinicore/ops/deepseek_v4_c4_act_quant_fused_scale.hpp"

#include "deepseek_v4_c4_act_quant_fused_scale_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4C4ActQuantFusedScaleKernel);

namespace {

constexpr int64_t kC4IndexerHeadDim = 128;

void check_hygon_tensor(const Tensor &tensor, const char *op_name) {
#if defined(ENABLE_HYGON_API)
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
#else
    (void)tensor;
    (void)op_name;
#endif
}

int c4_scalar_type_for_kernel(const Tensor &tensor, const char *op_name) {
    if (tensor->dtype() == DataType::BF16) {
        return deepseek_v4_c4_act_quant_fused_scale::kDsv4BF16;
    }
    if (tensor->dtype() == DataType::F16) {
        return deepseek_v4_c4_act_quant_fused_scale::kDsv4F16;
    }
    if (tensor->dtype() == DataType::F32) {
        return deepseek_v4_c4_act_quant_fused_scale::kDsv4F32;
    }
    throw std::runtime_error(std::string(op_name) + " supports bf16/fp16/fp32 tensors.");
}

void deepseek_v4_c4_act_quant_fused_scale_kernel_impl(const Tensor &q,
                                                      const Tensor &indexer_weights,
                                                      Tensor q_fp8,
                                                      Tensor q_scale,
                                                      Tensor fused_weights,
                                                      float weight_scale) {
#if defined(ENABLE_HYGON_API)
    constexpr const char *op_name = "deepseek_v4_c4_act_quant_fused_scale_kernel_";
    check_hygon_tensor(q, op_name);
    if (q->ndim() != 3 || q->size(2) != static_cast<size_t>(kC4IndexerHeadDim)) {
        throw std::runtime_error("deepseek_v4_c4_act_quant_fused_scale_kernel_ expects q [batch, heads, 128].");
    }
    if (indexer_weights->ndim() != 2 || indexer_weights->size(0) != q->size(0) || indexer_weights->size(1) != q->size(1)) {
        throw std::runtime_error("deepseek_v4_c4_act_quant_fused_scale_kernel_ expects weights [batch, heads].");
    }
    if (q_fp8->shape() != q->shape() || q_fp8->dtype() != DataType::F8) {
        throw std::runtime_error("deepseek_v4_c4_act_quant_fused_scale_kernel_ expects q_fp8 [batch, heads, 128] dtype f8.");
    }
    if (q_scale->ndim() != 3 || q_scale->size(0) != q->size(0) || q_scale->size(1) != q->size(1) || q_scale->size(2) != 1 || q_scale->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_c4_act_quant_fused_scale_kernel_ expects q_scale [batch, heads, 1] fp32.");
    }
    if (fused_weights->shape() != indexer_weights->shape() || fused_weights->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_c4_act_quant_fused_scale_kernel_ expects fused_weights [batch, heads] fp32.");
    }
    if (!q->is_contiguous() || !indexer_weights->is_contiguous() || !q_fp8->is_contiguous() ||
        !q_scale->is_contiguous() || !fused_weights->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_c4_act_quant_fused_scale_kernel_ expects contiguous tensors.");
    }
    deepseek_v4_c4_act_quant_fused_scale::launch_c4_act_quant_fused_scale(
        q->data(),
        c4_scalar_type_for_kernel(q, op_name),
        indexer_weights->data(),
        c4_scalar_type_for_kernel(indexer_weights, op_name),
        reinterpret_cast<uint8_t *>(q_fp8->data()),
        reinterpret_cast<float *>(q_scale->data()),
        reinterpret_cast<float *>(fused_weights->data()),
        static_cast<int64_t>(q->size(0) * q->size(1)),
        weight_scale,
        context::getStream());
#else
    (void)q;
    (void)indexer_weights;
    (void)q_fp8;
    (void)q_scale;
    (void)fused_weights;
    (void)weight_scale;
    throw std::runtime_error("deepseek_v4_c4_act_quant_fused_scale_kernel_ requires a HYGON build.");
#endif
}

} // namespace

DeepseekV4C4ActQuantFusedScaleKernel::DeepseekV4C4ActQuantFusedScaleKernel(const Tensor &q,
                                                                           const Tensor &indexer_weights,
                                                                           Tensor q_fp8,
                                                                           Tensor q_scale,
                                                                           Tensor fused_weights,
                                                                           float weight_scale) {
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(), q, indexer_weights, q_fp8, q_scale, fused_weights, weight_scale);
}

void DeepseekV4C4ActQuantFusedScaleKernel::execute(const Tensor &q,
                                                   const Tensor &indexer_weights,
                                                   Tensor q_fp8,
                                                   Tensor q_scale,
                                                   Tensor fused_weights,
                                                   float weight_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4C4ActQuantFusedScaleKernel, q, indexer_weights, q_fp8, q_scale, fused_weights, weight_scale);
}

namespace deepseek_v4_c4_act_quant_fused_scale_graph_impl {

struct PlannedMeta {
    graph::GraphTensor q;
    graph::GraphTensor indexer_weights;
    graph::GraphTensor q_fp8;
    graph::GraphTensor q_scale;
    graph::GraphTensor fused_weights;
    float weight_scale;
};

void *plan(const Tensor &q,
           const Tensor &indexer_weights,
           Tensor q_fp8,
           Tensor q_scale,
           Tensor fused_weights,
           float weight_scale) {
    return new PlannedMeta{graph::GraphTensor(q),
                           graph::GraphTensor(indexer_weights),
                           graph::GraphTensor(q_fp8),
                           graph::GraphTensor(q_scale),
                           graph::GraphTensor(fused_weights),
                           weight_scale};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_c4_act_quant_fused_scale_kernel_impl(planned->q,
                                                     planned->indexer_weights,
                                                     planned->q_fp8,
                                                     planned->q_scale,
                                                     planned->fused_weights,
                                                     planned->weight_scale);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_c4_act_quant_fused_scale_graph_impl

namespace deepseek_v4_c4_act_quant_fused_scale_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4C4ActQuantFusedScaleKernel,
                                       &deepseek_v4_c4_act_quant_fused_scale_graph_impl::plan,
                                       &deepseek_v4_c4_act_quant_fused_scale_graph_impl::run,
                                       &deepseek_v4_c4_act_quant_fused_scale_graph_impl::cleanup);
} // namespace deepseek_v4_c4_act_quant_fused_scale_register

void deepseek_v4_c4_act_quant_fused_scale_kernel_(const Tensor &q,
                                                  const Tensor &indexer_weights,
                                                  Tensor q_fp8,
                                                  Tensor q_scale,
                                                  Tensor fused_weights,
                                                  float weight_scale) {
    DeepseekV4C4ActQuantFusedScaleKernel::execute(q, indexer_weights, q_fp8, q_scale, fused_weights, weight_scale);
}

} // namespace infinicore::op
