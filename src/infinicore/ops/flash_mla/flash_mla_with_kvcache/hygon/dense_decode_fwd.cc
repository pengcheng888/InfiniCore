#include "infinicore/ops/flash_mla/flash_mla_with_kvcache.hpp"

#include "dense_decode_symbol.hpp"
#include "infinicore/context/context.hpp"

#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#endif
#endif

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op {

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
namespace {

void check_device(const Tensor &tensor, const char *op_name) {
    if (!tensor || tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors.");
    }
}

void check_optional_device(const std::optional<Tensor> &tensor, const char *op_name) {
    if (tensor.has_value() && tensor.value()) {
        check_device(*tensor, op_name);
    }
}

bool has_tensor(const std::optional<Tensor> &tensor) {
    return tensor.has_value() && tensor.value();
}

void check_hygon_dense_decode_options(const std::optional<Tensor> &block_table,
                                      const std::optional<Tensor> &cache_seqlens,
                                      const std::optional<Tensor> &num_splits,
                                      bool is_fp8_kvcache,
                                      const std::optional<Tensor> &indices,
                                      const std::optional<Tensor> &attn_sink,
                                      const std::optional<Tensor> &extra_k_cache,
                                      const std::optional<Tensor> &extra_indices_in_kvcache,
                                      const std::optional<Tensor> &topk_length,
                                      const std::optional<Tensor> &extra_topk_length,
                                      const char *op_name) {
    if (!has_tensor(block_table) || !has_tensor(cache_seqlens)) {
        throw std::runtime_error(std::string(op_name) + " requires block_table and cache_seqlens on HYGON dense decode.");
    }
    if (is_fp8_kvcache) {
        throw std::runtime_error(std::string(op_name) + " does not support is_fp8_kvcache=true on HYGON dense decode.");
    }
    if (has_tensor(num_splits)) {
        throw std::runtime_error(std::string(op_name) + " does not support the num_splits override on HYGON dense decode.");
    }
    if (has_tensor(indices)
        || has_tensor(attn_sink)
        || has_tensor(extra_k_cache)
        || has_tensor(extra_indices_in_kvcache)
        || has_tensor(topk_length)
        || has_tensor(extra_topk_length)) {
        throw std::runtime_error(std::string(op_name) + " currently supports dense attention only on HYGON.");
    }
}

double resolve_softmax_scale(const Tensor &q,
                             const std::optional<double> &softmax_scale,
                             const char *op_name) {
    if (softmax_scale.has_value()) {
        return softmax_scale.value();
    }
    if (!q || q->ndim() == 0 || q->size(q->ndim() - 1) == 0) {
        throw std::runtime_error(std::string(op_name) + " cannot infer softmax_scale from q.");
    }
    return 1.0 / std::sqrt(static_cast<double>(q->size(q->ndim() - 1)));
}

DataType from_at_scalar_type_for_dense_decode(at::ScalarType dtype) {
    switch (dtype) {
    case at::kFloat:
        return DataType::F32;
    case at::kHalf:
        return DataType::F16;
    case at::kBFloat16:
        return DataType::BF16;
    case at::kChar:
        return DataType::I8;
    case at::kInt:
        return DataType::I32;
    case at::kLong:
        return DataType::I64;
    case at::kByte:
        return DataType::U8;
    case at::kFloat8_e4m3fnuz:
        return DataType::F8;
    default:
        throw std::runtime_error("flash_mla_with_kvcache_impl: unsupported FlashMLA return dtype.");
    }
}

Device from_at_device_for_dense_decode(const at::Device &device) {
    if (device.is_cpu()) {
        return Device(Device::Type::CPU, 0);
    }
    if (!device.is_cuda()) {
        throw std::runtime_error("flash_mla_with_kvcache_impl: unsupported FlashMLA return device.");
    }
    return Device(Device::Type::HYGON, static_cast<Device::Index>(device.index()));
}

Shape shape_from_at_tensor_for_dense_decode(const at::Tensor &tensor) {
    Shape shape;
    shape.reserve(static_cast<size_t>(tensor.dim()));
    for (const auto dim : tensor.sizes()) {
        shape.push_back(static_cast<size_t>(dim));
    }
    return shape;
}

void copy_flashmla_return_tensor_exact(Tensor &dst, at::Tensor src, const char *name) {
    if (!src.defined()) {
        throw std::runtime_error(std::string("flash_mla_with_kvcache_impl: FlashMLA returned undefined ") + name + ".");
    }
    src = src.contiguous();
    const auto expected_shape = shape_from_at_tensor_for_dense_decode(src);
    const auto expected_dtype = from_at_scalar_type_for_dense_decode(src.scalar_type());
    const auto expected_device = from_at_device_for_dense_decode(src.device());
    if (!dst) {
        throw std::runtime_error(std::string("flash_mla_with_kvcache_impl: ") + name + " output must be preallocated.");
    }
    if (dst->shape() != expected_shape) {
        throw std::runtime_error(std::string("flash_mla_with_kvcache_impl: ") + name + " shape mismatch.");
    }
    if (dst->dtype() != expected_dtype) {
        throw std::runtime_error(std::string("flash_mla_with_kvcache_impl: ") + name + " dtype mismatch.");
    }
    if (dst->device() != expected_device) {
        throw std::runtime_error(std::string("flash_mla_with_kvcache_impl: ") + name + " device mismatch.");
    }
    if (!dst->is_contiguous()) {
        throw std::runtime_error(std::string("flash_mla_with_kvcache_impl: ") + name + " must be contiguous.");
    }
    auto dst_at = infinicore::adaptor::to_aten_tensor(dst);
    dst_at.copy_(src);
}

at::Tensor to_aten_tensor_for_flashmla(const Tensor &tensor) {
    if (tensor->dtype() == DataType::F8) {
        std::vector<int64_t> sizes(tensor->shape().begin(), tensor->shape().end());
        std::vector<int64_t> strides(tensor->strides().begin(), tensor->strides().end());
        auto options = at::TensorOptions()
                           .dtype(at::ScalarType::Float8_e4m3fn)
                           .device(infinicore::adaptor::to_at_device(tensor->device()))
                           .requires_grad(false);
        auto *data = const_cast<std::byte *>(tensor->data());
        return at::from_blob(
            data,
            sizes,
            strides,
            [](void *) {},
            options);
    }
    return infinicore::adaptor::to_aten_tensor(tensor);
}

std::optional<graph::GraphTensor> to_optional_graph_tensor(const std::optional<Tensor> &tensor) {
    if (!has_tensor(tensor)) {
        return std::nullopt;
    }
    return graph::GraphTensor(tensor.value());
}

std::optional<Tensor> to_optional_tensor(const std::optional<graph::GraphTensor> &tensor) {
    if (!tensor.has_value()) {
        return std::nullopt;
    }
    return tensor.value();
}

} // namespace

namespace flash_mla::flash_mla_with_kvcache_hygon {

void flash_mla_with_kvcache_impl(
    Tensor &out,
    Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> block_table,
    std::optional<Tensor> cache_seqlens,
    int64_t head_dim_v,
    const FlashMLASchedMeta &tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    std::optional<double> softmax_scale,
    bool causal,
    bool is_fp8_kvcache,
    std::optional<Tensor> indices,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> extra_topk_length);

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor lse;
    graph::GraphTensor q;
    graph::GraphTensor k_cache;
    std::optional<graph::GraphTensor> block_table;
    std::optional<graph::GraphTensor> cache_seqlens;
    int64_t head_dim_v;
    graph::GraphTensor tile_scheduler_metadata;
    graph::GraphTensor scheduler_num_splits;
    std::optional<graph::GraphTensor> num_splits;
    std::optional<double> softmax_scale;
    bool causal;
    bool is_fp8_kvcache;
    std::optional<graph::GraphTensor> indices;
    std::optional<graph::GraphTensor> attn_sink;
    std::optional<graph::GraphTensor> extra_k_cache;
    std::optional<graph::GraphTensor> extra_indices_in_kvcache;
    std::optional<graph::GraphTensor> topk_length;
    std::optional<graph::GraphTensor> extra_topk_length;
};

void *plan(Tensor out,
           Tensor lse,
           const Tensor &q,
           const Tensor &k_cache,
           std::optional<Tensor> block_table,
           std::optional<Tensor> cache_seqlens,
           int64_t head_dim_v,
           const FlashMLASchedMeta &tile_scheduler_metadata,
           std::optional<Tensor> num_splits,
           std::optional<double> softmax_scale,
           bool causal,
           bool is_fp8_kvcache,
           std::optional<Tensor> indices,
           std::optional<Tensor> attn_sink,
           std::optional<Tensor> extra_k_cache,
           std::optional<Tensor> extra_indices_in_kvcache,
           std::optional<Tensor> topk_length,
           std::optional<Tensor> extra_topk_length) {
    check_hygon_dense_decode_options(block_table,
                                     cache_seqlens,
                                     num_splits,
                                     is_fp8_kvcache,
                                     indices,
                                     attn_sink,
                                     extra_k_cache,
                                     extra_indices_in_kvcache,
                                     topk_length,
                                     extra_topk_length,
                                     "FlashMlaWithKvcache::plan");
    if (!tile_scheduler_metadata.has_sched_buffer()) {
        throw std::runtime_error("FlashMlaWithKvcache::plan requires precomputed scheduler metadata.");
    }
    return new PlannedMeta{graph::GraphTensor(out),
                           graph::GraphTensor(lse),
                           graph::GraphTensor(q),
                           graph::GraphTensor(k_cache),
                           to_optional_graph_tensor(block_table),
                           to_optional_graph_tensor(cache_seqlens),
                           head_dim_v,
                           graph::GraphTensor(tile_scheduler_metadata.tile_scheduler_metadata),
                           graph::GraphTensor(tile_scheduler_metadata.num_splits),
                           to_optional_graph_tensor(num_splits),
                           softmax_scale,
                           causal,
                           is_fp8_kvcache,
                           to_optional_graph_tensor(indices),
                           to_optional_graph_tensor(attn_sink),
                           to_optional_graph_tensor(extra_k_cache),
                           to_optional_graph_tensor(extra_indices_in_kvcache),
                           to_optional_graph_tensor(topk_length),
                           to_optional_graph_tensor(extra_topk_length)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    FlashMLASchedMeta scheduler_metadata;
    scheduler_metadata.tile_scheduler_metadata = planned->tile_scheduler_metadata;
    scheduler_metadata.num_splits = planned->scheduler_num_splits;
    flash_mla_with_kvcache_impl(planned->out,
                                planned->lse,
                                planned->q,
                                planned->k_cache,
                                to_optional_tensor(planned->block_table),
                                to_optional_tensor(planned->cache_seqlens),
                                planned->head_dim_v,
                                scheduler_metadata,
                                to_optional_tensor(planned->num_splits),
                                planned->softmax_scale,
                                planned->causal,
                                planned->is_fp8_kvcache,
                                to_optional_tensor(planned->indices),
                                to_optional_tensor(planned->attn_sink),
                                to_optional_tensor(planned->extra_k_cache),
                                to_optional_tensor(planned->extra_indices_in_kvcache),
                                to_optional_tensor(planned->topk_length),
                                to_optional_tensor(planned->extra_topk_length));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::flash_mla::FlashMlaWithKvcache::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    ::infinicore::op::flash_mla::FlashMlaWithKvcache::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    ::infinicore::op::flash_mla::FlashMlaWithKvcache::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    ::infinicore::op::flash_mla::flash_mla_with_kvcache_impl_dispatcher().registerDevice(Device::Type::HYGON, &flash_mla_with_kvcache_impl);
    return true;
}();

void flash_mla_with_kvcache_impl(
    Tensor &out,
    Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> block_table,
    std::optional<Tensor> cache_seqlens,
    int64_t head_dim_v,
    const FlashMLASchedMeta &tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    std::optional<double> softmax_scale,
    bool causal,
    bool is_fp8_kvcache,
    std::optional<Tensor> indices,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> extra_topk_length) {
    constexpr const char *op_name = "flash_mla_with_kvcache_impl";
    check_hygon_dense_decode_options(block_table,
                                     cache_seqlens,
                                     num_splits,
                                     is_fp8_kvcache,
                                     indices,
                                     attn_sink,
                                     extra_k_cache,
                                     extra_indices_in_kvcache,
                                     topk_length,
                                     extra_topk_length,
                                     op_name);
    const Tensor &sched_tile_metadata = tile_scheduler_metadata.tile_scheduler_metadata;
    const Tensor &sched_num_splits = tile_scheduler_metadata.num_splits;
    if (static_cast<bool>(sched_tile_metadata) != static_cast<bool>(sched_num_splits)) {
        throw std::runtime_error(std::string(op_name) + " expects scheduler metadata and scheduler num_splits to both be set or both be empty.");
    }

    const Tensor &block_table_tensor = block_table.value();
    const Tensor &cache_seqlens_tensor = cache_seqlens.value();
    const double scale = resolve_softmax_scale(q, softmax_scale, op_name);

    check_device(out, op_name);
    check_device(lse, op_name);
    check_device(q, op_name);
    check_device(k_cache, op_name);
    check_device(block_table_tensor, op_name);
    check_device(cache_seqlens_tensor, op_name);
    if (sched_tile_metadata) {
        check_device(sched_tile_metadata, op_name);
        check_device(sched_num_splits, op_name);
    }
    check_optional_device(indices, op_name);
    check_optional_device(attn_sink, op_name);
    check_optional_device(extra_k_cache, op_name);
    check_optional_device(extra_indices_in_kvcache, op_name);
    check_optional_device(topk_length, op_name);
    check_optional_device(extra_topk_length, op_name);

    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    auto q_flash_at = to_aten_tensor_for_flashmla(q);
    auto k_cache_flash_at = to_aten_tensor_for_flashmla(k_cache);
    auto cache_seqlens_flash_at = to_aten_tensor_for_flashmla(cache_seqlens_tensor);
    auto block_table_flash_at = to_aten_tensor_for_flashmla(block_table_tensor);
    std::optional<at::Tensor> tile_scheduler_metadata_flash_at;
    std::optional<at::Tensor> num_splits_flash_at;
    if (sched_tile_metadata) {
        tile_scheduler_metadata_flash_at = to_aten_tensor_for_flashmla(sched_tile_metadata);
        num_splits_flash_at = to_aten_tensor_for_flashmla(sched_num_splits);
    }
    auto [flash_out_at, flash_lse_at, new_tile_scheduler_metadata, new_num_splits] =
        flashmla_dense_decode_fn(op_name)(q_flash_at,
                                         k_cache_flash_at,
                                         static_cast<int>(head_dim_v),
                                         cache_seqlens_flash_at,
                                         block_table_flash_at,
                                         static_cast<float>(scale),
                                         causal,
                                         tile_scheduler_metadata_flash_at,
                                         num_splits_flash_at);
    // Scheduler metadata reuse is intentionally disabled by the public eager path.
    (void)new_tile_scheduler_metadata;
    (void)new_num_splits;
    copy_flashmla_return_tensor_exact(out, flash_out_at, "out");
    copy_flashmla_return_tensor_exact(lse, flash_lse_at, "lse");
}

} // namespace flash_mla::flash_mla_with_kvcache_hygon
#endif

} // namespace infinicore::op
