#include "infinicore/ops/deepseek_v4_fused_k_norm_rope_flashmla.hpp"

#include "deepseek_v4_fused_k_norm_rope_flashmla_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4FusedKNormRopeFlashMLAKernel);

namespace {

constexpr int64_t kHeadDim = 512;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kValueBytesPerToken = 576;
constexpr int64_t kScaleBytesPerToken = 8;

int64_t div_ceil_i64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

int64_t flashmla_page_bytes(int page_size) {
    const auto bytes = (kValueBytesPerToken + kScaleBytesPerToken) * static_cast<int64_t>(page_size);
    return div_ceil_i64(bytes, kValueBytesPerToken) * kValueBytesPerToken;
}

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

void check_fused_k_norm_rope_flashmla_shapes(const Tensor &kv,
                                             const Tensor &kv_weight,
                                             const Tensor &freqs_cis,
                                             const Tensor &positions,
                                             const Tensor &out_loc,
                                             const Tensor &kvcache,
                                             int page_size,
                                             const char *op_name) {
    check_accel_tensor(kv, op_name);
    check_accel_tensor(kv_weight, op_name);
    check_accel_tensor(freqs_cis, op_name);
    check_accel_tensor(positions, op_name);
    check_accel_tensor(out_loc, op_name);
    check_accel_tensor(kvcache, op_name);

    if (kv->ndim() != 2 || kv->size(1) != static_cast<size_t>(kHeadDim)) {
        throw std::runtime_error(std::string(op_name) + " expects kv [tokens, 512].");
    }
    if (kv->dtype() != DataType::BF16 || kv_weight->dtype() != DataType::BF16) {
        throw std::runtime_error(std::string(op_name) + " supports bf16 kv/kv_weight only.");
    }
    if (kv->stride(1) != 1 || kv->stride(0) < static_cast<size_t>(kHeadDim)) {
        throw std::runtime_error(std::string(op_name) + " expects row-strided kv [tokens, 512] with contiguous last dimension.");
    }
    if (kv_weight->ndim() != 1 || kv_weight->size(0) != static_cast<size_t>(kHeadDim) || !kv_weight->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous kv_weight [512].");
    }
    if (freqs_cis->ndim() != 2 || freqs_cis->size(1) != static_cast<size_t>(kRopeDim) || freqs_cis->dtype() != DataType::F32 || !freqs_cis->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous freqs_cis [max_pos, 64] float32.");
    }
    if (positions->ndim() != 1 || positions->size(0) != kv->size(0) || (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64) || !positions->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous positions [tokens] int32/int64.");
    }
    if (out_loc->ndim() != 1 || out_loc->size(0) != kv->size(0) || (out_loc->dtype() != DataType::I32 && out_loc->dtype() != DataType::I64) || !out_loc->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous out_loc [tokens] int32/int64.");
    }
    if (kvcache->ndim() != 2 || kvcache->dtype() != DataType::U8 || !kvcache->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous uint8 kvcache [pages, page_bytes].");
    }
    if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
        throw std::runtime_error(std::string(op_name) + " expects page_size to be a positive power of two.");
    }
    if (kvcache->size(1) != static_cast<size_t>(flashmla_page_bytes(page_size))) {
        throw std::runtime_error(std::string(op_name) + " kvcache page_bytes mismatch.");
    }
}

} // namespace

DeepseekV4FusedKNormRopeFlashMLAKernel::DeepseekV4FusedKNormRopeFlashMLAKernel(const Tensor &kv,
                                                                               const Tensor &kv_weight,
                                                                               float epsilon,
                                                                               const Tensor &freqs_cis,
                                                                               const Tensor &positions,
                                                                               const Tensor &out_loc,
                                                                               Tensor kvcache,
                                                                               int page_size) {
    INFINICORE_GRAPH_OP_DISPATCH(kv->device().getType(),
                                 kv,
                                 kv_weight,
                                 epsilon,
                                 freqs_cis,
                                 positions,
                                 out_loc,
                                 kvcache,
                                 page_size);
    return;
}

void DeepseekV4FusedKNormRopeFlashMLAKernel::execute(const Tensor &kv,
                                                     const Tensor &kv_weight,
                                                     float epsilon,
                                                     const Tensor &freqs_cis,
                                                     const Tensor &positions,
                                                     const Tensor &out_loc,
                                                     Tensor kvcache,
                                                     int page_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4FusedKNormRopeFlashMLAKernel,
                                      kv,
                                      kv_weight,
                                      epsilon,
                                      freqs_cis,
                                      positions,
                                      out_loc,
                                      kvcache,
                                      page_size);
    return;
}

namespace deepseek_v4_fused_k_norm_rope_flashmla_graph_impl {

struct PlannedMeta {
    graph::GraphTensor kv;
    graph::GraphTensor kv_weight;
    graph::GraphTensor freqs_cis;
    graph::GraphTensor positions;
    graph::GraphTensor out_loc;
    graph::GraphTensor kvcache;
    bool positions_i64;
    bool out_loc_i64;
    int64_t tokens;
    int64_t kv_stride_batch;
    int page_size;
    int64_t page_bytes;
    float epsilon;
};

void *plan(const Tensor &kv,
           const Tensor &kv_weight,
           float epsilon,
           const Tensor &freqs_cis,
           const Tensor &positions,
           const Tensor &out_loc,
           Tensor kvcache,
           int page_size) {
    check_fused_k_norm_rope_flashmla_shapes(kv,
                                            kv_weight,
                                            freqs_cis,
                                            positions,
                                            out_loc,
                                            kvcache,
                                            page_size,
                                            "deepseek_v4_fused_k_norm_rope_flashmla_kernel_");
    return new PlannedMeta{graph::GraphTensor(kv),
                           graph::GraphTensor(kv_weight),
                           graph::GraphTensor(freqs_cis),
                           graph::GraphTensor(positions),
                           graph::GraphTensor(out_loc),
                           graph::GraphTensor(kvcache),
                           positions->dtype() == DataType::I64,
                           out_loc->dtype() == DataType::I64,
                           static_cast<int64_t>(kv->size(0)),
                           static_cast<int64_t>(kv->stride(0)),
                           page_size,
                           flashmla_page_bytes(page_size),
                           epsilon};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_fused_k_norm_rope_flashmla_native::launch_fused_k_norm_rope_flashmla(
        planned->kv->data(),
        planned->kv_weight->data(),
        reinterpret_cast<const float *>(planned->freqs_cis->data()),
        planned->positions->data(),
        planned->positions_i64,
        planned->out_loc->data(),
        planned->out_loc_i64,
        reinterpret_cast<uint8_t *>(planned->kvcache->data()),
        planned->tokens,
        planned->kv_stride_batch,
        planned->page_size,
        planned->page_bytes,
        planned->epsilon,
        context::getStream());
    return;
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_fused_k_norm_rope_flashmla_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
    return;
}

} // namespace deepseek_v4_fused_k_norm_rope_flashmla_graph_impl

namespace deepseek_v4_fused_k_norm_rope_flashmla_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4FusedKNormRopeFlashMLAKernel,
                                       &deepseek_v4_fused_k_norm_rope_flashmla_graph_impl::plan,
                                       &deepseek_v4_fused_k_norm_rope_flashmla_graph_impl::run,
                                       &deepseek_v4_fused_k_norm_rope_flashmla_graph_impl::cleanup);
} // namespace deepseek_v4_fused_k_norm_rope_flashmla_register

void deepseek_v4_fused_k_norm_rope_flashmla_kernel_(const Tensor &kv,
                                                    const Tensor &kv_weight,
                                                    float epsilon,
                                                    const Tensor &freqs_cis,
                                                    const Tensor &positions,
                                                    const Tensor &out_loc,
                                                    Tensor kvcache,
                                                    int page_size) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    check_fused_k_norm_rope_flashmla_shapes(kv,
                                            kv_weight,
                                            freqs_cis,
                                            positions,
                                            out_loc,
                                            kvcache,
                                            page_size,
                                            "deepseek_v4_fused_k_norm_rope_flashmla_kernel_");
    DeepseekV4FusedKNormRopeFlashMLAKernel::execute(kv, kv_weight, epsilon, freqs_cis, positions, out_loc, kvcache, page_size);
    return;
#else
    (void)kv;
    (void)kv_weight;
    (void)epsilon;
    (void)freqs_cis;
    (void)positions;
    (void)out_loc;
    (void)kvcache;
    (void)page_size;
    throw std::runtime_error("deepseek_v4_fused_k_norm_rope_flashmla_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_fused_k_norm_rope_flashmla_(const Tensor &kv,
                                             const Tensor &kv_weight,
                                             float epsilon,
                                             const Tensor &freqs_cis,
                                             const Tensor &positions,
                                             const Tensor &out_loc,
                                             Tensor kvcache,
                                             int page_size) {
    deepseek_v4_fused_k_norm_rope_flashmla_kernel_(kv,
                                                   kv_weight,
                                                   epsilon,
                                                   freqs_cis,
                                                   positions,
                                                   out_loc,
                                                   kvcache,
                                                   page_size);
    return;
}

} // namespace infinicore::op
