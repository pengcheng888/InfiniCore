#include "infinicore/ops/deepseek_v4_fused_q_norm_rope.hpp"

#include "deepseek_v4_fused_q_norm_rope_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace {

constexpr size_t kDsv4HeadDim = 512;
constexpr size_t kDsv4RopeDim = 64;

void check_accel_device(const Tensor &tensor, const char *op_name) {
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

void check_contiguous_tensor(const Tensor &tensor, const char *op_name) {
    check_accel_device(tensor, op_name);
    if (!tensor->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
}

void check_common_freqs_positions(const Tensor &freqs_cis,
                                  const Tensor &positions,
                                  size_t tokens,
                                  const char *op_name) {
    check_contiguous_tensor(freqs_cis, op_name);
    check_contiguous_tensor(positions, op_name);
    if (freqs_cis->ndim() != 2
        || freqs_cis->size(1) != kDsv4RopeDim
        || freqs_cis->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " expects freqs_cis [max_pos, 64] float32.");
    }
    if (positions->ndim() != 1
        || positions->size(0) != tokens
        || (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64)) {
        throw std::runtime_error(std::string(op_name) + " expects positions [tokens] int32/int64.");
    }
}

void check_q_norm_rope_layout(const Tensor &tensor, const char *tensor_name, const char *op_name) {
    check_accel_device(tensor, op_name);
    if (tensor->stride(2) != 1 || tensor->stride(1) != static_cast<Stride>(kDsv4HeadDim)) {
        throw std::runtime_error(std::string(op_name) + " expects " + tensor_name + " strides [*, 512, 1].");
    }
}

void check_q_norm_rope_shapes(const Tensor &q_out,
                              const Tensor &q_input,
                              const Tensor &freqs_cis,
                              const Tensor &positions,
                              const char *op_name) {
    if (q_input->ndim() != 3 || q_input->size(2) != kDsv4HeadDim) {
        throw std::runtime_error(std::string(op_name) + " expects q_input [tokens, heads, 512].");
    }
    if (q_out->shape() != q_input->shape()) {
        throw std::runtime_error(std::string(op_name) + " q_out shape mismatch.");
    }
    if (q_input->dtype() != DataType::BF16
        || q_out->dtype() != DataType::BF16) {
        throw std::runtime_error(std::string(op_name) + " supports bf16 q_input/q_out only.");
    }
    if (q_out->device().getType() != q_input->device().getType()
        || q_out->device().getIndex() != q_input->device().getIndex()) {
        throw std::runtime_error(std::string(op_name) + " q_out/q_input device mismatch.");
    }
    check_q_norm_rope_layout(q_input, "q_input", op_name);
    check_q_norm_rope_layout(q_out, "q_out", op_name);
    check_common_freqs_positions(freqs_cis, positions, q_input->size(0), op_name);
}

} // namespace

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FusedQNormRope);

FusedQNormRope::FusedQNormRope(Tensor q_out,
                               const Tensor &q_input,
                               float epsilon,
                               const Tensor &freqs_cis,
                               const Tensor &positions) {
    INFINICORE_GRAPH_OP_DISPATCH(q_out->device().getType(),
                                 q_out, q_input, epsilon, freqs_cis, positions);
}

void FusedQNormRope::execute(Tensor q_out,
                             const Tensor &q_input,
                             float epsilon,
                             const Tensor &freqs_cis,
                             const Tensor &positions) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(FusedQNormRope,
                                      q_out, q_input, epsilon, freqs_cis, positions);
}

namespace deepseek_v4_fused_q_norm_rope_impl {

struct PlannedMeta {
    graph::GraphTensor q_out;
    graph::GraphTensor q_input;
    graph::GraphTensor freqs_cis;
    graph::GraphTensor positions;
    bool positions_i64;
    int64_t tokens;
    int64_t heads;
    int64_t q_input_stride_batch;
    int64_t q_out_stride_batch;
    float epsilon;
};

void *plan(Tensor q_out,
           const Tensor &q_input,
           float epsilon,
           const Tensor &freqs_cis,
           const Tensor &positions) {
    check_q_norm_rope_shapes(q_out, q_input, freqs_cis, positions, "deepseek_v4_fused_q_norm_rope_kernel_");
    return new PlannedMeta{graph::GraphTensor(q_out),
                           graph::GraphTensor(q_input),
                           graph::GraphTensor(freqs_cis),
                           graph::GraphTensor(positions),
                           positions->dtype() == DataType::I64,
                           static_cast<int64_t>(q_input->size(0)),
                           static_cast<int64_t>(q_input->size(1)),
                           static_cast<int64_t>(q_input->stride(0)),
                           static_cast<int64_t>(q_out->stride(0)),
                           epsilon};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    ::infinicore::op::deepseek_v4_fused_q_norm_rope::launch_fused_q_norm_rope(
        planned->q_out->data(),
        planned->q_input->data(),
        reinterpret_cast<const float *>(planned->freqs_cis->data()),
        planned->positions->data(),
        planned->positions_i64,
        planned->tokens,
        planned->heads,
        planned->q_input_stride_batch,
        planned->q_out_stride_batch,
        planned->epsilon,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_fused_q_norm_rope_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_fused_q_norm_rope_impl

namespace deepseek_v4_fused_q_norm_rope_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(FusedQNormRope,
                                       &deepseek_v4_fused_q_norm_rope_impl::plan,
                                       &deepseek_v4_fused_q_norm_rope_impl::run,
                                       &deepseek_v4_fused_q_norm_rope_impl::cleanup);
} // namespace deepseek_v4_fused_q_norm_rope_register

} // namespace deepseek_v4

void deepseek_v4_fused_q_norm_rope_kernel_(Tensor q_out,
                                           const Tensor &q_input,
                                           float epsilon,
                                           const Tensor &freqs_cis,
                                           const Tensor &positions) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    check_q_norm_rope_shapes(q_out, q_input, freqs_cis, positions, "deepseek_v4_fused_q_norm_rope_kernel_");
    deepseek_v4::FusedQNormRope::execute(q_out, q_input, epsilon, freqs_cis, positions);
#else
    (void)q_out;
    (void)q_input;
    (void)epsilon;
    (void)freqs_cis;
    (void)positions;
    throw std::runtime_error("deepseek_v4_fused_q_norm_rope_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
