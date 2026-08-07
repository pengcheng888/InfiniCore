#include "infinicore/ops/deepseek_v4_fused_norm_rope_inplace.hpp"

#include "deepseek_v4_fused_norm_rope_inplace_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4FusedNormRopeInplaceKernel);

namespace {

constexpr size_t kDsv4HeadDim = 512;
constexpr size_t kDsv4RopeDim = 64;

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
    if (!tensor->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
}

void check_common_freqs_positions(const Tensor &freqs_cis,
                                  const Tensor &positions,
                                  size_t tokens,
                                  const char *op_name) {
    check_accel_tensor(freqs_cis, op_name);
    check_accel_tensor(positions, op_name);
    if (freqs_cis->ndim() != 2 || freqs_cis->size(1) != kDsv4RopeDim || freqs_cis->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " expects freqs_cis [max_pos, 64] float32.");
    }
    if (positions->ndim() != 1 || positions->size(0) != tokens ||
        (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64)) {
        throw std::runtime_error(std::string(op_name) + " expects positions [tokens] int32/int64.");
    }
}

void check_norm_rope_inplace_shapes(const Tensor &input,
                                    const Tensor &norm_weight,
                                    const Tensor &freqs_cis,
                                    const Tensor &positions,
                                    const char *op_name) {
    check_accel_tensor(input, op_name);
    check_accel_tensor(norm_weight, op_name);
    if (input->ndim() != 2 || input->size(1) != kDsv4HeadDim) {
        throw std::runtime_error(std::string(op_name) + " expects input [tokens, 512].");
    }
    if (input->dtype() != DataType::BF16 || norm_weight->dtype() != DataType::BF16) {
        throw std::runtime_error(std::string(op_name) + " supports bf16 input/norm_weight only.");
    }
    if (norm_weight->ndim() != 1 || norm_weight->size(0) != kDsv4HeadDim) {
        throw std::runtime_error(std::string(op_name) + " expects norm_weight [512].");
    }
    check_common_freqs_positions(freqs_cis, positions, input->size(0), op_name);
}

} // namespace

DeepseekV4FusedNormRopeInplaceKernel::DeepseekV4FusedNormRopeInplaceKernel(Tensor input,
                                                                           const Tensor &norm_weight,
                                                                           float epsilon,
                                                                           const Tensor &freqs_cis,
                                                                           const Tensor &positions) {
    INFINICORE_GRAPH_OP_DISPATCH(input->device().getType(), input, norm_weight, epsilon, freqs_cis, positions);
}

void DeepseekV4FusedNormRopeInplaceKernel::execute(Tensor input,
                                                   const Tensor &norm_weight,
                                                   float epsilon,
                                                   const Tensor &freqs_cis,
                                                   const Tensor &positions) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4FusedNormRopeInplaceKernel, input, norm_weight, epsilon, freqs_cis, positions);
}

namespace deepseek_v4_fused_norm_rope_inplace_graph_impl {

struct PlannedMeta {
    graph::GraphTensor input;
    graph::GraphTensor norm_weight;
    graph::GraphTensor freqs_cis;
    graph::GraphTensor positions;
    bool positions_i64;
    int64_t tokens;
    float epsilon;
};

void *plan(Tensor input,
           const Tensor &norm_weight,
           float epsilon,
           const Tensor &freqs_cis,
           const Tensor &positions) {
    check_norm_rope_inplace_shapes(input, norm_weight, freqs_cis, positions, "deepseek_v4_fused_norm_rope_inplace_kernel_");
    return new PlannedMeta{graph::GraphTensor(input),
                           graph::GraphTensor(norm_weight),
                           graph::GraphTensor(freqs_cis),
                           graph::GraphTensor(positions),
                           positions->dtype() == DataType::I64,
                           static_cast<int64_t>(input->size(0)),
                           epsilon};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_fused_norm_rope_inplace_native::launch_fused_norm_rope_inplace(planned->input->data(),
                                                                               planned->norm_weight->data(),
                                                                               reinterpret_cast<const float *>(planned->freqs_cis->data()),
                                                                               planned->positions->data(),
                                                                               planned->positions_i64,
                                                                               planned->tokens,
                                                                               planned->epsilon,
                                                                               context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_fused_norm_rope_inplace_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_fused_norm_rope_inplace_graph_impl

namespace deepseek_v4_fused_norm_rope_inplace_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4FusedNormRopeInplaceKernel,
                                       &deepseek_v4_fused_norm_rope_inplace_graph_impl::plan,
                                       &deepseek_v4_fused_norm_rope_inplace_graph_impl::run,
                                       &deepseek_v4_fused_norm_rope_inplace_graph_impl::cleanup);
} // namespace deepseek_v4_fused_norm_rope_inplace_register

void deepseek_v4_fused_norm_rope_inplace_kernel_(Tensor input,
                                                 const Tensor &norm_weight,
                                                 float epsilon,
                                                 const Tensor &freqs_cis,
                                                 const Tensor &positions) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    check_norm_rope_inplace_shapes(input, norm_weight, freqs_cis, positions, "deepseek_v4_fused_norm_rope_inplace_kernel_");
    DeepseekV4FusedNormRopeInplaceKernel::execute(input, norm_weight, epsilon, freqs_cis, positions);
#else
    (void)input;
    (void)norm_weight;
    (void)epsilon;
    (void)freqs_cis;
    (void)positions;
    throw std::runtime_error("deepseek_v4_fused_norm_rope_inplace_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_fused_norm_rope_inplace_(Tensor input,
                                          const Tensor &norm_weight,
                                          float epsilon,
                                          const Tensor &freqs_cis,
                                          const Tensor &positions) {
    deepseek_v4_fused_norm_rope_inplace_kernel_(input, norm_weight, epsilon, freqs_cis, positions);
}

void deepseek_v4_fused_norm_rope_inplace(Tensor input,
                                         const Tensor &norm_weight,
                                         float epsilon,
                                         const Tensor &freqs_cis,
                                         const Tensor &positions) {
    deepseek_v4_fused_norm_rope_inplace_(input, norm_weight, epsilon, freqs_cis, positions);
}

} // namespace infinicore::op
