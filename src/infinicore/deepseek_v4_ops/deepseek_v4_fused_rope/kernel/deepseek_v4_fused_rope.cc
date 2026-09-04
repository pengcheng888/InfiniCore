#include "infinicore/ops/deepseek_v4_fused_rope.hpp"

#include "deepseek_v4_fused_rope_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace infinicore::op {

namespace {

constexpr size_t kDsv4RopeDim = 64;

void check_tensor_device(const Tensor &tensor, const char *op_name) {
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

void check_rope_tensor(const Tensor &tensor, const char *name) {
    if (tensor->ndim() != 2 && tensor->ndim() != 3) {
        throw std::runtime_error(std::string("deepseek_v4_fused_rope_kernel_ expects ") + name + " rank 2 or 3.");
    }
    if (tensor->size(tensor->ndim() - 1) != kDsv4RopeDim) {
        throw std::runtime_error(std::string("deepseek_v4_fused_rope_kernel_ expects ") + name + " last dim 64.");
    }
    if (tensor->dtype() != DataType::BF16) {
        throw std::runtime_error(std::string("deepseek_v4_fused_rope_kernel_ expects ") + name + " dtype bf16.");
    }
    if (tensor->stride(tensor->ndim() - 1) != 1) {
        throw std::runtime_error(std::string("deepseek_v4_fused_rope_kernel_ expects ") + name + " last-dim stride 1.");
    }
}

void check_common_inputs(const Tensor &query,
                         const std::optional<Tensor> &key,
                         const Tensor &freqs_cis,
                         const Tensor &positions,
                         const char *op_name) {
    check_tensor_device(query, op_name);
    check_tensor_device(freqs_cis, op_name);
    check_tensor_device(positions, op_name);
    check_rope_tensor(query, "query");
    if (key.has_value()) {
        check_tensor_device(key.value(), op_name);
        check_rope_tensor(key.value(), "key");
        if (key.value()->size(0) != query->size(0)) {
            throw std::runtime_error(std::string(op_name) + " key/query batch mismatch.");
        }
    }
    if (freqs_cis->ndim() != 2 || freqs_cis->size(1) != kDsv4RopeDim || freqs_cis->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " expects freqs_cis [max_pos, 64] float32.");
    }
    if (positions->ndim() != 1 || (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64)) {
        throw std::runtime_error(std::string(op_name) + " expects positions [tokens] int32/int64.");
    }
    if (positions->size(0) != query->size(0)) {
        throw std::runtime_error(std::string(op_name) + " positions length mismatch.");
    }
}

int dsv4_scalar_type_for_kernel(const Tensor &tensor, const char *op_name) {
    if (tensor->dtype() == DataType::BF16) {
        return deepseek_v4_fused_rope_kernel::kDsv4BF16;
    }
    if (tensor->dtype() == DataType::F16) {
        return deepseek_v4_fused_rope_kernel::kDsv4F16;
    }
    if (tensor->dtype() == DataType::F32) {
        return deepseek_v4_fused_rope_kernel::kDsv4F32;
    }
    throw std::runtime_error(std::string(op_name) + " supports bf16/fp16/fp32 tensors only.");
}

} // namespace

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FusedRope);

FusedRope::FusedRope(Tensor query,
                     std::optional<Tensor> key,
                     const Tensor &freqs_cis,
                     const Tensor &positions,
                     bool inverse) {
    INFINICORE_GRAPH_OP_DISPATCH(query->device().getType(), query, key, freqs_cis, positions, inverse);
}

void FusedRope::execute(Tensor query,
                        std::optional<Tensor> key,
                        const Tensor &freqs_cis,
                        const Tensor &positions,
                        bool inverse) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(FusedRope, query, key, freqs_cis, positions, inverse);
}

namespace deepseek_v4_fused_rope_impl {

struct PlannedTensorMeta {
    graph::GraphTensor tensor;
    int dtype;
    int64_t tokens;
    int64_t heads;
    int64_t stride_token;
    int64_t stride_head;
};

struct PlannedMeta {
    PlannedTensorMeta query;
    std::optional<PlannedTensorMeta> key;
    graph::GraphTensor freqs_cis;
    graph::GraphTensor positions;
    bool positions_i64;
    bool inverse;
};

PlannedTensorMeta make_tensor_meta(const Tensor &tensor, const char *op_name) {
    return {graph::GraphTensor(tensor),
            dsv4_scalar_type_for_kernel(tensor, op_name),
            static_cast<int64_t>(tensor->size(0)),
            tensor->ndim() == 3 ? static_cast<int64_t>(tensor->size(1)) : 1,
            static_cast<int64_t>(tensor->stride(0)),
            tensor->ndim() == 3 ? static_cast<int64_t>(tensor->stride(1)) : 0};
}

void *plan(Tensor query,
           std::optional<Tensor> key,
           const Tensor &freqs_cis,
           const Tensor &positions,
           bool inverse) {
    check_common_inputs(query, key, freqs_cis, positions, "deepseek_v4_fused_rope_kernel_");
    std::optional<PlannedTensorMeta> key_meta = std::nullopt;
    if (key.has_value()) {
        key_meta = make_tensor_meta(key.value(), "deepseek_v4_fused_rope_kernel_");
    }
    return new PlannedMeta{make_tensor_meta(query, "deepseek_v4_fused_rope_kernel_"),
                           std::move(key_meta),
                           graph::GraphTensor(freqs_cis),
                           graph::GraphTensor(positions),
                           positions->dtype() == DataType::I64,
                           inverse};
}

void run_one(PlannedTensorMeta &meta,
             const graph::GraphTensor &freqs_cis,
             const graph::GraphTensor &positions,
             bool positions_i64,
             bool inverse) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    deepseek_v4_fused_rope_kernel::launch_fused_rope(meta.tensor->data(),
                                                     meta.dtype,
                                                     freqs_cis->data(),
                                                     positions->data(),
                                                     positions_i64,
                                                     meta.tokens,
                                                     meta.heads,
                                                     meta.stride_token,
                                                     meta.stride_head,
                                                     inverse,
                                                     context::getStream());
#else
    (void)meta;
    (void)freqs_cis;
    (void)positions;
    (void)positions_i64;
    (void)inverse;
    throw std::runtime_error("deepseek_v4_fused_rope_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    run_one(planned->query, planned->freqs_cis, planned->positions, planned->positions_i64, planned->inverse);
    if (planned->key.has_value()) {
        run_one(planned->key.value(), planned->freqs_cis, planned->positions, planned->positions_i64, planned->inverse);
    }
#elif defined(ENABLE_ATEN) && defined(ENABLE_METAX_API)
    std::optional<Tensor> key = std::nullopt;
    if (planned->key.has_value()) {
        key = planned->key.value().tensor;
    }
    deepseek_v4_fused_rope_aten_(
        planned->query.tensor,
        key,
        planned->freqs_cis,
        planned->positions,
        planned->inverse);
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_fused_rope_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_fused_rope_impl

namespace deepseek_v4_fused_rope_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(FusedRope,
                                       &deepseek_v4_fused_rope_impl::plan,
                                       &deepseek_v4_fused_rope_impl::run,
                                       &deepseek_v4_fused_rope_impl::cleanup);
} // namespace deepseek_v4_fused_rope_register

} // namespace deepseek_v4

void deepseek_v4_fused_rope_kernel_(Tensor query,
                                    std::optional<Tensor> key,
                                    const Tensor &freqs_cis,
                                    const Tensor &positions,
                                    bool inverse) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    check_common_inputs(query, key, freqs_cis, positions, "deepseek_v4_fused_rope_kernel_");
    deepseek_v4::FusedRope::execute(query, key, freqs_cis, positions, inverse);
#else
    (void)query;
    (void)key;
    (void)freqs_cis;
    (void)positions;
    (void)inverse;
    throw std::runtime_error("deepseek_v4_fused_rope_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op
