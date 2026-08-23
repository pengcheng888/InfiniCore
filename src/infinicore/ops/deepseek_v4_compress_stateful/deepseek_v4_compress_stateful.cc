#include "infinicore/ops/deepseek_v4_compress_stateful.hpp"

#include "deepseek_v4_compress_stateful_kernel.hpp"

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

int c4_ape_layout(const Tensor &ape, int64_t head_dim, const char *op_name) {
    if (ape->ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " expects ape rank 2.");
    }
    if (ape->size(0) == 8 && ape->size(1) == static_cast<size_t>(head_dim)) {
        return 0;
    }
    if (ape->size(0) == 4 && ape->size(1) == static_cast<size_t>(2 * head_dim)) {
        return 1;
    }
    throw std::runtime_error(std::string(op_name) + " expects ape [8, head_dim] or [4, 2 * head_dim].");
}

} // namespace

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4C4CompressStatefulKernel);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4C128CompressStatefulKernel);

DeepseekV4C4CompressStatefulKernel::DeepseekV4C4CompressStatefulKernel(
    Tensor output,
    const Tensor &kv_score_input,
    const Tensor &ape,
    Tensor compressor_state,
    const Tensor &write_loc,
    const Tensor &extra_loc,
    const Tensor &positions) {
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(),
                                 output,
                                 kv_score_input,
                                 ape,
                                 compressor_state,
                                 write_loc,
                                 extra_loc,
                                 positions);
}

void DeepseekV4C4CompressStatefulKernel::execute(Tensor output,
                                                 const Tensor &kv_score_input,
                                                 const Tensor &ape,
                                                 Tensor compressor_state,
                                                 const Tensor &write_loc,
                                                 const Tensor &extra_loc,
                                                 const Tensor &positions) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4C4CompressStatefulKernel,
                                      output,
                                      kv_score_input,
                                      ape,
                                      compressor_state,
                                      write_loc,
                                      extra_loc,
                                      positions);
}

namespace deepseek_v4_c4_compress_stateful_graph_impl {

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor kv_score_input;
    graph::GraphTensor ape;
    graph::GraphTensor compressor_state;
    graph::GraphTensor write_loc;
    graph::GraphTensor extra_loc;
    graph::GraphTensor positions;
    Tensor output_owner;
    int output_dtype;
    int kv_score_dtype;
    int state_dtype;
    int ape_dtype;
    int ape_layout;
    bool write_loc_i64;
    bool extra_loc_i64;
    bool positions_i64;
    int64_t extra_cols;
    int64_t tokens;
    int64_t head_dim;
};

void *plan(Tensor output,
           const Tensor &kv_score_input,
           const Tensor &ape,
           Tensor compressor_state,
           const Tensor &write_loc,
           const Tensor &extra_loc,
           const Tensor &positions) {
    check_common_accel_tensor(kv_score_input, "DeepseekV4C4CompressStatefulKernel");
    check_common_accel_tensor(ape, "DeepseekV4C4CompressStatefulKernel");
    check_common_accel_tensor(compressor_state, "DeepseekV4C4CompressStatefulKernel");
    check_common_accel_tensor(write_loc, "DeepseekV4C4CompressStatefulKernel");
    check_common_accel_tensor(extra_loc, "DeepseekV4C4CompressStatefulKernel");
    check_common_accel_tensor(positions, "DeepseekV4C4CompressStatefulKernel");
    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 4 != 0) {
        throw std::runtime_error("DeepseekV4C4CompressStatefulKernel expects kv_score_input [tokens, 4 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 4);
    if (output->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}) {
        throw std::runtime_error("DeepseekV4C4CompressStatefulKernel output shape mismatch.");
    }
    if (compressor_state->ndim() != 2 || compressor_state->size(1) != static_cast<size_t>(4 * head_dim) || compressor_state->size(0) % 4 != 0) {
        throw std::runtime_error("DeepseekV4C4CompressStatefulKernel compressor_state shape mismatch.");
    }
    if (write_loc->numel() != static_cast<size_t>(tokens) || positions->numel() != static_cast<size_t>(tokens)) {
        throw std::runtime_error("DeepseekV4C4CompressStatefulKernel metadata token count mismatch.");
    }
    int64_t extra_cols = 1;
    if (extra_loc->ndim() == 2) {
        if (extra_loc->size(0) != static_cast<size_t>(tokens) || extra_loc->size(1) < 1) {
            throw std::runtime_error("DeepseekV4C4CompressStatefulKernel expects extra_loc [tokens, >=1].");
        }
        extra_cols = static_cast<int64_t>(extra_loc->size(1));
    } else if (extra_loc->ndim() != 1 || extra_loc->size(0) != static_cast<size_t>(tokens)) {
        throw std::runtime_error("DeepseekV4C4CompressStatefulKernel expects extra_loc rank 1 or 2.");
    }
    const int ape_layout = c4_ape_layout(ape, head_dim, "DeepseekV4C4CompressStatefulKernel");
    return new PlannedMeta{graph::GraphTensor(output),
                           graph::GraphTensor(kv_score_input),
                           graph::GraphTensor(ape),
                           graph::GraphTensor(compressor_state),
                           graph::GraphTensor(write_loc),
                           graph::GraphTensor(extra_loc),
                           graph::GraphTensor(positions),
                           output,
                           dsv4_scalar_type_for_kernel(output, "DeepseekV4C4CompressStatefulKernel"),
                           dsv4_scalar_type_for_kernel(kv_score_input, "DeepseekV4C4CompressStatefulKernel"),
                           dsv4_scalar_type_for_kernel(compressor_state, "DeepseekV4C4CompressStatefulKernel"),
                           dsv4_scalar_type_for_kernel(ape, "DeepseekV4C4CompressStatefulKernel"),
                           ape_layout,
                           write_loc->dtype() == DataType::I64,
                           extra_loc->dtype() == DataType::I64,
                           positions->dtype() == DataType::I64,
                           extra_cols,
                           tokens,
                           head_dim};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_compress_stateful_kernel::launch_c4_compress_stateful(
        planned->output->data(),
        planned->output_dtype,
        planned->kv_score_input->data(),
        planned->kv_score_dtype,
        planned->compressor_state->data(),
        planned->state_dtype,
        planned->ape->data(),
        planned->ape_dtype,
        planned->ape_layout,
        planned->write_loc->data(),
        planned->write_loc_i64,
        planned->extra_loc->data(),
        planned->extra_loc_i64,
        planned->extra_cols,
        planned->positions->data(),
        planned->positions_i64,
        planned->tokens,
        planned->head_dim,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("DeepseekV4C4CompressStatefulKernel requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_c4_compress_stateful_graph_impl

namespace deepseek_v4_c4_compress_stateful_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4C4CompressStatefulKernel,
                                       &deepseek_v4_c4_compress_stateful_graph_impl::plan,
                                       &deepseek_v4_c4_compress_stateful_graph_impl::run,
                                       &deepseek_v4_c4_compress_stateful_graph_impl::cleanup);
} // namespace deepseek_v4_c4_compress_stateful_register

DeepseekV4C128CompressStatefulKernel::DeepseekV4C128CompressStatefulKernel(
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

void DeepseekV4C128CompressStatefulKernel::execute(Tensor output,
                                                   const Tensor &kv_score_input,
                                                   const Tensor &ape,
                                                   Tensor compressor_state,
                                                   const Tensor &write_loc,
                                                   const Tensor &positions) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4C128CompressStatefulKernel,
                                      output,
                                      kv_score_input,
                                      ape,
                                      compressor_state,
                                      write_loc,
                                      positions);
}

namespace deepseek_v4_c128_compress_stateful_graph_impl {

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
    check_common_accel_tensor(kv_score_input, "DeepseekV4C128CompressStatefulKernel");
    check_common_accel_tensor(ape, "DeepseekV4C128CompressStatefulKernel");
    check_common_accel_tensor(compressor_state, "DeepseekV4C128CompressStatefulKernel");
    check_common_accel_tensor(write_loc, "DeepseekV4C128CompressStatefulKernel");
    check_common_accel_tensor(positions, "DeepseekV4C128CompressStatefulKernel");
    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 2 != 0) {
        throw std::runtime_error("DeepseekV4C128CompressStatefulKernel expects kv_score_input [tokens, 2 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 2);
    if (head_dim != kDsv4FlashMlaQDim) {
        throw std::runtime_error("DeepseekV4C128CompressStatefulKernel expects head_dim 512.");
    }
    if (output->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}) {
        throw std::runtime_error("DeepseekV4C128CompressStatefulKernel output shape mismatch.");
    }
    if (compressor_state->ndim() != 2 || compressor_state->size(1) != static_cast<size_t>(2 * head_dim) || compressor_state->size(0) % 128 != 0) {
        throw std::runtime_error("DeepseekV4C128CompressStatefulKernel compressor_state shape mismatch.");
    }
    if (write_loc->numel() != static_cast<size_t>(tokens) || positions->numel() != static_cast<size_t>(tokens)) {
        throw std::runtime_error("DeepseekV4C128CompressStatefulKernel metadata token count mismatch.");
    }
    if (ape->ndim() != 2 || ape->size(0) < 128 || ape->size(1) != static_cast<size_t>(head_dim)) {
        throw std::runtime_error("DeepseekV4C128CompressStatefulKernel expects ape [>=128, head_dim].");
    }
    return new PlannedMeta{graph::GraphTensor(output),
                           graph::GraphTensor(kv_score_input),
                           graph::GraphTensor(ape),
                           graph::GraphTensor(compressor_state),
                           graph::GraphTensor(write_loc),
                           graph::GraphTensor(positions),
                           output,
                           dsv4_scalar_type_for_kernel(output, "DeepseekV4C128CompressStatefulKernel"),
                           dsv4_scalar_type_for_kernel(kv_score_input, "DeepseekV4C128CompressStatefulKernel"),
                           dsv4_scalar_type_for_kernel(compressor_state, "DeepseekV4C128CompressStatefulKernel"),
                           dsv4_scalar_type_for_kernel(ape, "DeepseekV4C128CompressStatefulKernel"),
                           write_loc->dtype() == DataType::I64,
                           positions->dtype() == DataType::I64,
                           tokens,
                           head_dim};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_compress_stateful_kernel::launch_c128_compress_stateful(
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
    throw std::runtime_error("DeepseekV4C128CompressStatefulKernel requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_c128_compress_stateful_graph_impl

namespace deepseek_v4_c128_compress_stateful_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4C128CompressStatefulKernel,
                                       &deepseek_v4_c128_compress_stateful_graph_impl::plan,
                                       &deepseek_v4_c128_compress_stateful_graph_impl::run,
                                       &deepseek_v4_c128_compress_stateful_graph_impl::cleanup);
} // namespace deepseek_v4_c128_compress_stateful_register

Tensor deepseek_v4_c4_compress_stateful_kernel(const Tensor &kv_score_input,
                                               const Tensor &ape,
                                               Tensor compressor_state,
                                               const Tensor &write_loc,
                                               const Tensor &extra_loc,
                                               const Tensor &positions) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_common_accel_tensor(kv_score_input, "deepseek_v4_c4_compress_stateful_kernel");
    check_common_accel_tensor(ape, "deepseek_v4_c4_compress_stateful_kernel");
    check_common_accel_tensor(compressor_state, "deepseek_v4_c4_compress_stateful_kernel");
    check_common_accel_tensor(write_loc, "deepseek_v4_c4_compress_stateful_kernel");
    check_common_accel_tensor(extra_loc, "deepseek_v4_c4_compress_stateful_kernel");
    check_common_accel_tensor(positions, "deepseek_v4_c4_compress_stateful_kernel");
    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 4 != 0) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel expects kv_score_input [tokens, 4 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 4);
    if (compressor_state->ndim() != 2 || compressor_state->size(1) != static_cast<size_t>(4 * head_dim) || compressor_state->size(0) % 4 != 0) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel expects compressor_state [4 * groups, 4 * head_dim].");
    }
    if (write_loc->numel() != static_cast<size_t>(tokens) || positions->numel() != static_cast<size_t>(tokens)) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel metadata token count mismatch.");
    }
    if (write_loc->dtype() != DataType::I32 && write_loc->dtype() != DataType::I64) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel write_loc must be int32/int64.");
    }
    if (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel positions must be int32/int64.");
    }
    if (extra_loc->dtype() != DataType::I32 && extra_loc->dtype() != DataType::I64) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel extra_loc must be int32/int64.");
    }
    int64_t extra_cols = 1;
    if (extra_loc->ndim() == 2) {
        if (extra_loc->size(0) != static_cast<size_t>(tokens) || extra_loc->size(1) < 1) {
            throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel expects extra_loc [tokens, >=1].");
        }
        extra_cols = static_cast<int64_t>(extra_loc->size(1));
    } else if (extra_loc->ndim() != 1 || extra_loc->size(0) != static_cast<size_t>(tokens)) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel expects extra_loc rank 1 or 2.");
    }
    const int ape_layout = c4_ape_layout(ape, head_dim, "deepseek_v4_c4_compress_stateful_kernel");
    auto output = Tensor::empty({static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}, kv_score_input->dtype(), kv_score_input->device());
    DeepseekV4C4CompressStatefulKernel::execute(output,
                                                kv_score_input,
                                                ape,
                                                compressor_state,
                                                write_loc,
                                                extra_loc,
                                                positions);
    return output;
#else
    (void)kv_score_input;
    (void)ape;
    (void)compressor_state;
    (void)write_loc;
    (void)extra_loc;
    (void)positions;
    throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

Tensor deepseek_v4_c4_compress_stateful(const Tensor &kv_score_input,
                                        const Tensor &ape,
                                        Tensor compressor_state,
                                        const Tensor &write_loc,
                                        const Tensor &extra_loc,
                                        const Tensor &positions) {
    return deepseek_v4_c4_compress_stateful_kernel(kv_score_input,
                                                   ape,
                                                   compressor_state,
                                                   write_loc,
                                                   extra_loc,
                                                   positions);
}

Tensor deepseek_v4_c128_compress_stateful_kernel(const Tensor &kv_score_input,
                                                 const Tensor &ape,
                                                 Tensor compressor_state,
                                                 const Tensor &write_loc,
                                                 const Tensor &positions) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_common_accel_tensor(kv_score_input, "deepseek_v4_c128_compress_stateful_kernel");
    check_common_accel_tensor(ape, "deepseek_v4_c128_compress_stateful_kernel");
    check_common_accel_tensor(compressor_state, "deepseek_v4_c128_compress_stateful_kernel");
    check_common_accel_tensor(write_loc, "deepseek_v4_c128_compress_stateful_kernel");
    check_common_accel_tensor(positions, "deepseek_v4_c128_compress_stateful_kernel");
    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 2 != 0) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel expects kv_score_input [tokens, 2 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 2);
    if (head_dim != kDsv4FlashMlaQDim) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel expects head_dim 512.");
    }
    if (compressor_state->ndim() != 2 || compressor_state->size(1) != static_cast<size_t>(2 * head_dim) || compressor_state->size(0) % 128 != 0) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel expects compressor_state [128 * groups, 2 * head_dim].");
    }
    if (write_loc->numel() != static_cast<size_t>(tokens) || positions->numel() != static_cast<size_t>(tokens)) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel metadata token count mismatch.");
    }
    if (write_loc->dtype() != DataType::I32 && write_loc->dtype() != DataType::I64) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel write_loc must be int32/int64.");
    }
    if (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel positions must be int32/int64.");
    }
    if (ape->ndim() != 2 || ape->size(0) < 128 || ape->size(1) != static_cast<size_t>(head_dim)) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel expects ape [>=128, head_dim].");
    }
    auto output = Tensor::empty({static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}, kv_score_input->dtype(), kv_score_input->device());
    DeepseekV4C128CompressStatefulKernel::execute(output,
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
    throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

Tensor deepseek_v4_c128_compress_stateful(const Tensor &kv_score_input,
                                          const Tensor &ape,
                                          Tensor compressor_state,
                                          const Tensor &write_loc,
                                          const Tensor &positions) {
    return deepseek_v4_c128_compress_stateful_kernel(kv_score_input,
                                                     ape,
                                                     compressor_state,
                                                     write_loc,
                                                     positions);
}


} // namespace infinicore::op
