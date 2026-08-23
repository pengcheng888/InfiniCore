#include "infinicore/ops/deepseek_v4_compress_sglang_stateful.hpp"

#include "deepseek_v4_c128_compress_sglang_stateful_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace {

constexpr int64_t kDsv4FlashMlaQDim = 512;
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

} // namespace

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4C128CompressSglangStatefulKernel);

DeepseekV4C128CompressSglangStatefulKernel::DeepseekV4C128CompressSglangStatefulKernel(
    Tensor output,
    const Tensor &kv_score_input,
    const Tensor &ape,
    Tensor compressor_state,
    const Tensor &write_loc,
    const Tensor &positions) {
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(),
                                 output,
                                 kv_score_input,
                                 ape,
                                 compressor_state,
                                 write_loc,
                                 positions);
}

void DeepseekV4C128CompressSglangStatefulKernel::execute(Tensor output,
                                                         const Tensor &kv_score_input,
                                                         const Tensor &ape,
                                                         Tensor compressor_state,
                                                         const Tensor &write_loc,
                                                         const Tensor &positions) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4C128CompressSglangStatefulKernel,
                                      output,
                                      kv_score_input,
                                      ape,
                                      compressor_state,
                                      write_loc,
                                      positions);
}

namespace deepseek_v4_c128_compress_sglang_stateful_impl {

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor kv_score_input;
    graph::GraphTensor ape;
    graph::GraphTensor compressor_state;
    graph::GraphTensor write_loc;
    graph::GraphTensor positions;
    Tensor output_owner;
    int output_dtype;
    int kv_score_dtype;
    int state_dtype;
    int ape_dtype;
    bool write_loc_i64;
    bool positions_i64;
    int64_t tokens;
    int64_t head_dim;
};

void *plan(Tensor output,
           const Tensor &kv_score_input,
           const Tensor &ape,
           Tensor compressor_state,
           const Tensor &write_loc,
           const Tensor &positions) {
    check_common_accel_tensor(kv_score_input, "DeepseekV4C128CompressSglangStatefulKernel");
    check_common_accel_tensor(ape, "DeepseekV4C128CompressSglangStatefulKernel");
    check_common_accel_tensor(compressor_state, "DeepseekV4C128CompressSglangStatefulKernel");
    check_common_accel_tensor(write_loc, "DeepseekV4C128CompressSglangStatefulKernel");
    check_common_accel_tensor(positions, "DeepseekV4C128CompressSglangStatefulKernel");
    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 2 != 0) {
        throw std::runtime_error("DeepseekV4C128CompressSglangStatefulKernel expects kv_score_input [tokens, 2 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 2);
    if (head_dim != kDsv4FlashMlaQDim) {
        throw std::runtime_error("DeepseekV4C128CompressSglangStatefulKernel expects head_dim 512.");
    }
    if (output->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}) {
        throw std::runtime_error("DeepseekV4C128CompressSglangStatefulKernel output shape mismatch.");
    }
    if (compressor_state->ndim() != 2 || compressor_state->size(1) != static_cast<size_t>(2 * head_dim) || compressor_state->size(0) % 128 != 0) {
        throw std::runtime_error("DeepseekV4C128CompressSglangStatefulKernel compressor_state shape mismatch.");
    }
    if (write_loc->numel() != static_cast<size_t>(tokens) || positions->numel() != static_cast<size_t>(tokens)) {
        throw std::runtime_error("DeepseekV4C128CompressSglangStatefulKernel metadata token count mismatch.");
    }
    if (ape->ndim() != 2 || ape->size(0) < 128 || ape->size(1) != static_cast<size_t>(head_dim)) {
        throw std::runtime_error("DeepseekV4C128CompressSglangStatefulKernel expects ape [>=128, head_dim].");
    }
    return new PlannedMeta{graph::GraphTensor(output),
                           graph::GraphTensor(kv_score_input),
                           graph::GraphTensor(ape),
                           graph::GraphTensor(compressor_state),
                           graph::GraphTensor(write_loc),
                           graph::GraphTensor(positions),
                           output,
                           dsv4_scalar_type_for_kernel(output, "DeepseekV4C128CompressSglangStatefulKernel"),
                           dsv4_scalar_type_for_kernel(kv_score_input, "DeepseekV4C128CompressSglangStatefulKernel"),
                           dsv4_scalar_type_for_kernel(compressor_state, "DeepseekV4C128CompressSglangStatefulKernel"),
                           dsv4_scalar_type_for_kernel(ape, "DeepseekV4C128CompressSglangStatefulKernel"),
                           write_loc->dtype() == DataType::I64,
                           positions->dtype() == DataType::I64,
                           tokens,
                           head_dim};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_c128_compress_sglang_stateful_kernel_impl::launch_c128_compress_stateful_sglang(
        planned->output->data(),
        planned->output_dtype,
        planned->kv_score_input->data(),
        planned->kv_score_dtype,
        planned->compressor_state->data(),
        planned->state_dtype,
        planned->ape->data(),
        planned->ape_dtype,
        planned->write_loc->data(),
        planned->write_loc_i64,
        planned->positions->data(),
        planned->positions_i64,
        planned->tokens,
        planned->head_dim,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("DeepseekV4C128CompressSglangStatefulKernel requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_c128_compress_sglang_stateful_impl

namespace deepseek_v4_c128_compress_sglang_stateful_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4C128CompressSglangStatefulKernel,
                                       &deepseek_v4_c128_compress_sglang_stateful_impl::plan,
                                       &deepseek_v4_c128_compress_sglang_stateful_impl::run,
                                       &deepseek_v4_c128_compress_sglang_stateful_impl::cleanup);
} // namespace deepseek_v4_c128_compress_sglang_stateful_register

} // namespace deepseek_v4

Tensor deepseek_v4_c128_compress_sglang_stateful_kernel(const Tensor &kv_score_input,
                                                        const Tensor &ape,
                                                        Tensor compressor_state,
                                                        const Tensor &write_loc,
                                                        const Tensor &positions) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_common_accel_tensor(kv_score_input, "deepseek_v4_c128_compress_sglang_stateful_kernel");
    check_common_accel_tensor(ape, "deepseek_v4_c128_compress_sglang_stateful_kernel");
    check_common_accel_tensor(compressor_state, "deepseek_v4_c128_compress_sglang_stateful_kernel");
    check_common_accel_tensor(write_loc, "deepseek_v4_c128_compress_sglang_stateful_kernel");
    check_common_accel_tensor(positions, "deepseek_v4_c128_compress_sglang_stateful_kernel");
    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 2 != 0) {
        throw std::runtime_error("deepseek_v4_c128_compress_sglang_stateful_kernel expects kv_score_input [tokens, 2 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 2);
    if (head_dim != kDsv4FlashMlaQDim) {
        throw std::runtime_error("deepseek_v4_c128_compress_sglang_stateful_kernel expects head_dim 512.");
    }
    if (compressor_state->ndim() != 2 || compressor_state->size(1) != static_cast<size_t>(2 * head_dim) || compressor_state->size(0) % 128 != 0) {
        throw std::runtime_error("deepseek_v4_c128_compress_sglang_stateful_kernel expects compressor_state [128 * groups, 2 * head_dim].");
    }
    if (write_loc->numel() != static_cast<size_t>(tokens) || positions->numel() != static_cast<size_t>(tokens)) {
        throw std::runtime_error("deepseek_v4_c128_compress_sglang_stateful_kernel metadata token count mismatch.");
    }
    if (write_loc->dtype() != DataType::I32 && write_loc->dtype() != DataType::I64) {
        throw std::runtime_error("deepseek_v4_c128_compress_sglang_stateful_kernel write_loc must be int32/int64.");
    }
    if (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64) {
        throw std::runtime_error("deepseek_v4_c128_compress_sglang_stateful_kernel positions must be int32/int64.");
    }
    if (ape->ndim() != 2 || ape->size(0) < 128 || ape->size(1) != static_cast<size_t>(head_dim)) {
        throw std::runtime_error("deepseek_v4_c128_compress_sglang_stateful_kernel expects ape [>=128, head_dim].");
    }
    auto output = Tensor::empty({static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}, kv_score_input->dtype(), kv_score_input->device());
    deepseek_v4::DeepseekV4C128CompressSglangStatefulKernel::execute(output,
                                                                     kv_score_input,
                                                                     ape,
                                                                     compressor_state,
                                                                     write_loc,
                                                                     positions);
    return output;
#else
    (void)kv_score_input;
    (void)ape;
    (void)compressor_state;
    (void)write_loc;
    (void)positions;
    throw std::runtime_error("deepseek_v4_c128_compress_sglang_stateful_kernel requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
