#include "infinicore/ops/deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang.hpp"

#include "deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4FusedQIndexerRopeHadamardQuantSglangKernel);

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

void check_shapes(const Tensor &q_input,
                  const Tensor &q_fp8,
                  const Tensor &weight,
                  const Tensor &weights_out,
                  const Tensor &freqs_cis,
                  const Tensor &positions,
                  const char *op_name) {
    check_accel_tensor(q_input, op_name);
    check_accel_tensor(q_fp8, op_name);
    check_accel_tensor(weight, op_name);
    check_accel_tensor(weights_out, op_name);
    check_accel_tensor(freqs_cis, op_name);
    check_accel_tensor(positions, op_name);

    if (q_input->ndim() != 3 || q_input->size(2) != static_cast<size_t>(kHeadDim) || q_input->dtype() != DataType::BF16) {
        throw std::runtime_error(std::string(op_name) + " expects q_input contiguous [tokens, heads, 128] bf16.");
    }
    if (q_fp8->shape() != q_input->shape() || q_fp8->dtype() != DataType::F8) {
        throw std::runtime_error(std::string(op_name) + " expects q_fp8 [tokens, heads, 128] f8.");
    }
    if (weight->ndim() != 2 || weight->size(0) != q_input->size(0) || weight->size(1) != q_input->size(1) ||
        weight->dtype() != DataType::BF16) {
        throw std::runtime_error(std::string(op_name) + " expects weight [tokens, heads] bf16.");
    }
    if (weights_out->ndim() != 3 || weights_out->size(0) != q_input->size(0) || weights_out->size(1) != q_input->size(1) ||
        weights_out->size(2) != 1 || weights_out->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " expects weights_out [tokens, heads, 1] fp32.");
    }
    if (freqs_cis->ndim() != 2 || freqs_cis->size(1) != static_cast<size_t>(kRopeDim) || freqs_cis->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " expects freqs_cis [max_pos, 64] fp32.");
    }
    if (positions->ndim() != 1 || positions->size(0) != q_input->size(0) ||
        (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64)) {
        throw std::runtime_error(std::string(op_name) + " expects positions [tokens] int32/int64.");
    }
    if (!q_input->is_contiguous() || !q_fp8->is_contiguous() || !weight->is_contiguous() ||
        !weights_out->is_contiguous() || !freqs_cis->is_contiguous() || !positions->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
}

void fused_q_indexer_rope_hadamard_quant_sglang_impl(const Tensor &q_input,
                                                     Tensor q_fp8,
                                                     const Tensor &weight,
                                                     Tensor weights_out,
                                                     float weight_scale,
                                                     const Tensor &freqs_cis,
                                                     const Tensor &positions) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    constexpr const char *op_name = "deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_kernel_";
    check_shapes(q_input, q_fp8, weight, weights_out, freqs_cis, positions, op_name);
    if (q_input->size(0) == 0) {
        return;
    }
    deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang::launch_fused_q_indexer_rope_hadamard_quant_sglang(
        q_input->data(),
        reinterpret_cast<uint8_t *>(q_fp8->data()),
        weight->data(),
        reinterpret_cast<float *>(weights_out->data()),
        weight_scale,
        reinterpret_cast<const float *>(freqs_cis->data()),
        positions->data(),
        positions->dtype() == DataType::I64,
        static_cast<int64_t>(q_input->size(0) * q_input->size(1)),
        static_cast<int64_t>(q_input->size(1)),
        context::getStream());
#else
    (void)q_input;
    (void)q_fp8;
    (void)weight;
    (void)weights_out;
    (void)weight_scale;
    (void)freqs_cis;
    (void)positions;
    throw std::runtime_error("deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_kernel_ requires a HYGON/NVIDIA build.");
#endif
    return;
}

} // namespace

DeepseekV4FusedQIndexerRopeHadamardQuantSglangKernel::DeepseekV4FusedQIndexerRopeHadamardQuantSglangKernel(
    const Tensor &q_input,
    Tensor q_fp8,
    const Tensor &weight,
    Tensor weights_out,
    float weight_scale,
    const Tensor &freqs_cis,
    const Tensor &positions) {
    INFINICORE_GRAPH_OP_DISPATCH(q_input->device().getType(), q_input, q_fp8, weight, weights_out, weight_scale, freqs_cis, positions);
}

void DeepseekV4FusedQIndexerRopeHadamardQuantSglangKernel::execute(const Tensor &q_input,
                                                                   Tensor q_fp8,
                                                                   const Tensor &weight,
                                                                   Tensor weights_out,
                                                                   float weight_scale,
                                                                   const Tensor &freqs_cis,
                                                                   const Tensor &positions) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4FusedQIndexerRopeHadamardQuantSglangKernel, q_input, q_fp8, weight, weights_out, weight_scale, freqs_cis, positions);
    return;
}

namespace deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_graph_impl {

struct PlannedMeta {
    graph::GraphTensor q_input;
    graph::GraphTensor q_fp8;
    graph::GraphTensor weight;
    graph::GraphTensor weights_out;
    graph::GraphTensor freqs_cis;
    graph::GraphTensor positions;
    float weight_scale;
};

void *plan(const Tensor &q_input,
           Tensor q_fp8,
           const Tensor &weight,
           Tensor weights_out,
           float weight_scale,
           const Tensor &freqs_cis,
           const Tensor &positions) {
    return new PlannedMeta{graph::GraphTensor(q_input),
                           graph::GraphTensor(q_fp8),
                           graph::GraphTensor(weight),
                           graph::GraphTensor(weights_out),
                           graph::GraphTensor(freqs_cis),
                           graph::GraphTensor(positions),
                           weight_scale};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    fused_q_indexer_rope_hadamard_quant_sglang_impl(planned->q_input,
                                                    planned->q_fp8,
                                                    planned->weight,
                                                    planned->weights_out,
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

} // namespace deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_graph_impl

namespace deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4FusedQIndexerRopeHadamardQuantSglangKernel,
                                       &deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_graph_impl::plan,
                                       &deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_graph_impl::run,
                                       &deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_graph_impl::cleanup);
} // namespace deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_register

void deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_(const Tensor &q_input,
                                                             Tensor q_fp8,
                                                             const Tensor &weight,
                                                             Tensor weights_out,
                                                             float weight_scale,
                                                             const Tensor &freqs_cis,
                                                             const Tensor &positions) {
    deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_kernel_(q_input, q_fp8, weight, weights_out, weight_scale, freqs_cis, positions);
    return;
}

void deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_kernel_(const Tensor &q_input,
                                                                    Tensor q_fp8,
                                                                    const Tensor &weight,
                                                                    Tensor weights_out,
                                                                    float weight_scale,
                                                                    const Tensor &freqs_cis,
                                                                    const Tensor &positions) {
    DeepseekV4FusedQIndexerRopeHadamardQuantSglangKernel::execute(q_input, q_fp8, weight, weights_out, weight_scale, freqs_cis, positions);
    return;
}

} // namespace infinicore::op
