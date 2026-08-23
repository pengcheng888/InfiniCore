#include "infinicore/ops/deepseek_v4_compress_fused_norm_rope.hpp"

#include "deepseek_v4_compress_fused_norm_rope_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace {

constexpr int kDsv4BF16 = 0;
constexpr int kDsv4F16 = 1;
constexpr int kDsv4F32 = 2;

void check_hygon_or_nvidia_tensor(const Tensor &tensor, const char *op_name) {
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

int dsv4_scalar_type_for_kernel(const Tensor &tensor, const char *op_name) {
    if (tensor->dtype() == DataType::BF16) {
        return kDsv4BF16;
    }
    if (tensor->dtype() == DataType::F16) {
        return kDsv4F16;
    }
    if (tensor->dtype() == DataType::F32) {
        return kDsv4F32;
    }
    throw std::runtime_error(std::string(op_name) + " supports bf16/fp16/fp32 tensors only.");
}

void check_common_accel_tensor(const Tensor &tensor, const char *op_name) {
    check_hygon_or_nvidia_tensor(tensor, op_name);
    if (!tensor->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
}

void check_compress_fused_norm_rope_shapes(const Tensor &input,
                                           const Tensor &norm_weight,
                                           const Tensor &freqs_cis,
                                           const Tensor &positions) {
    if (input->ndim() != 2 || input->size(1) < 64) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_ expects input [tokens, dim>=64].");
    }
    if (input->dtype() != DataType::BF16) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_ expects bf16 input.");
    }
    if (norm_weight->numel() != input->size(1)) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_ norm_weight size mismatch.");
    }
    if (freqs_cis->ndim() != 2 || freqs_cis->size(1) != 64 || freqs_cis->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_ expects freqs_cis [max_pos, 64] float32.");
    }
    if (positions->ndim() != 1 || positions->numel() != input->size(0) || (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64)) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_ expects positions [tokens] int32/int64.");
    }
}

} // namespace

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4CompressFusedNormRopeKernel);

DeepseekV4CompressFusedNormRopeKernel::DeepseekV4CompressFusedNormRopeKernel(
    Tensor input,
    const Tensor &norm_weight,
    float epsilon,
    const Tensor &freqs_cis,
    const Tensor &positions) {
    INFINICORE_GRAPH_OP_DISPATCH(input->device().getType(), input, norm_weight, epsilon, freqs_cis, positions);
}

void DeepseekV4CompressFusedNormRopeKernel::execute(Tensor input,
                                                    const Tensor &norm_weight,
                                                    float epsilon,
                                                    const Tensor &freqs_cis,
                                                    const Tensor &positions) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4CompressFusedNormRopeKernel,
                                      input,
                                      norm_weight,
                                      epsilon,
                                      freqs_cis,
                                      positions);
}

namespace deepseek_v4_compress_fused_norm_rope_graph_impl {

struct PlannedMeta {
    graph::GraphTensor input;
    graph::GraphTensor norm_weight;
    graph::GraphTensor freqs_cis;
    graph::GraphTensor positions;
    int input_dtype;
    int norm_weight_dtype;
    bool positions_i64;
    int64_t tokens;
    int64_t dim;
    float epsilon;
};

void *plan(Tensor input,
           const Tensor &norm_weight,
           float epsilon,
           const Tensor &freqs_cis,
           const Tensor &positions) {
    check_compress_fused_norm_rope_shapes(input, norm_weight, freqs_cis, positions);
    check_common_accel_tensor(input, "DeepseekV4CompressFusedNormRopeKernel");
    check_common_accel_tensor(norm_weight, "DeepseekV4CompressFusedNormRopeKernel");
    check_common_accel_tensor(freqs_cis, "DeepseekV4CompressFusedNormRopeKernel");
    check_common_accel_tensor(positions, "DeepseekV4CompressFusedNormRopeKernel");
    return new PlannedMeta{graph::GraphTensor(input),
                           graph::GraphTensor(norm_weight),
                           graph::GraphTensor(freqs_cis),
                           graph::GraphTensor(positions),
                           dsv4_scalar_type_for_kernel(input, "DeepseekV4CompressFusedNormRopeKernel"),
                           dsv4_scalar_type_for_kernel(norm_weight, "DeepseekV4CompressFusedNormRopeKernel"),
                           positions->dtype() == DataType::I64,
                           static_cast<int64_t>(input->size(0)),
                           static_cast<int64_t>(input->size(1)),
                           epsilon};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_compress_fused_norm_rope_kernel::launch_compress_fused_norm_rope(
        planned->input->data(),
        planned->input_dtype,
        planned->norm_weight->data(),
        planned->norm_weight_dtype,
        reinterpret_cast<const float *>(planned->freqs_cis->data()),
        planned->positions->data(),
        planned->positions_i64,
        planned->tokens,
        planned->dim,
        planned->epsilon,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("DeepseekV4CompressFusedNormRopeKernel requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_compress_fused_norm_rope_graph_impl

namespace deepseek_v4_compress_fused_norm_rope_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4CompressFusedNormRopeKernel,
                                       &deepseek_v4_compress_fused_norm_rope_graph_impl::plan,
                                       &deepseek_v4_compress_fused_norm_rope_graph_impl::run,
                                       &deepseek_v4_compress_fused_norm_rope_graph_impl::cleanup);
} // namespace deepseek_v4_compress_fused_norm_rope_register

void deepseek_v4_compress_fused_norm_rope_kernel_(Tensor input,
                                                  const Tensor &norm_weight,
                                                  float epsilon,
                                                  const Tensor &freqs_cis,
                                                  const Tensor &positions) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_compress_fused_norm_rope_shapes(input, norm_weight, freqs_cis, positions);
    check_common_accel_tensor(input, "deepseek_v4_compress_fused_norm_rope_kernel_");
    check_common_accel_tensor(norm_weight, "deepseek_v4_compress_fused_norm_rope_kernel_");
    check_common_accel_tensor(freqs_cis, "deepseek_v4_compress_fused_norm_rope_kernel_");
    check_common_accel_tensor(positions, "deepseek_v4_compress_fused_norm_rope_kernel_");
    DeepseekV4CompressFusedNormRopeKernel::execute(input,
                                                   norm_weight,
                                                   epsilon,
                                                   freqs_cis,
                                                   positions);
#else
    (void)input;
    (void)norm_weight;
    (void)epsilon;
    (void)freqs_cis;
    (void)positions;
    throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_compress_fused_norm_rope_(Tensor input,
                                           const Tensor &norm_weight,
                                           float epsilon,
                                           const Tensor &freqs_cis,
                                           const Tensor &positions) {
    deepseek_v4_compress_fused_norm_rope_kernel_(input, norm_weight, epsilon, freqs_cis, positions);
}


} // namespace infinicore::op
