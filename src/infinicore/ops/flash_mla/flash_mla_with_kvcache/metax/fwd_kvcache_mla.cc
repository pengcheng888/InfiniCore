#include "infinicore/ops/flash_mla/flash_mla_with_kvcache.hpp"

#if defined(ENABLE_ATEN) && defined(ENABLE_METAX_API)

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op::flash_mla::flash_mla_with_kvcache_metax {
namespace {

using FlashMlaFwdKvcacheMlaFn = std::vector<at::Tensor> (*)(
    at::Tensor &,
    const at::Tensor &,
    std::optional<const at::Tensor> &,
    int,
    const at::Tensor &,
    const at::Tensor &,
    float,
    bool,
    const at::Tensor &,
    const at::Tensor &,
    bool,
    std::optional<const at::Tensor> &,
    std::optional<const at::Tensor> &,
    int,
    int,
    std::optional<const at::Tensor> &);

constexpr const char *kFlashMlaFwdKvcacheMlaSymbol = "_Z15fwd_kvcache_mlaRN2at6TensorERKS0_RSt8optionalIS2_EiS3_S3_fbS3_S3_bS6_S6_iiS6_";
constexpr const char *kDefaultFlashMlaSoPath = "/opt/conda/lib/python3.12/site-packages/flash_mla_cuda.cpython-312-x86_64-linux-gnu.so";

bool has_tensor(const std::optional<Tensor> &tensor) {
    return tensor.has_value() && tensor.value();
}

void check_device(const Tensor &tensor, const char *op_name) {
    if (!tensor || tensor->device().getType() != Device::Type::METAX) {
        throw std::runtime_error(std::string(op_name) + " expects METAX tensors.");
    }
}

void check_optional_device(const std::optional<Tensor> &tensor,
                           const char *op_name) {
    if (has_tensor(tensor)) {
        check_device(tensor.value(), op_name);
    }
}

void check_metax_options(const std::optional<Tensor> &block_table,
                         const std::optional<Tensor> &cache_seqlens,
                         const std::optional<Tensor> &num_splits,
                         bool causal,
                         const std::optional<Tensor> &indices,
                         const std::optional<Tensor> &attn_sink,
                         const std::optional<Tensor> &extra_k_cache,
                         const std::optional<Tensor> &extra_indices_in_kvcache,
                         const std::optional<Tensor> &topk_length,
                         const std::optional<Tensor> &extra_topk_length,
                         const char *op_name) {
    if (!has_tensor(block_table) || !has_tensor(cache_seqlens)) {
        throw std::runtime_error(
            std::string(op_name)
            + " requires block_table and cache_seqlens on METAX.");
    }
    if (has_tensor(num_splits)) {
        throw std::runtime_error(
            std::string(op_name)
            + " does not support the num_splits override on METAX.");
    }
    if (has_tensor(indices) && causal) {
        throw std::runtime_error(
            std::string(op_name)
            + " requires causal=false when sparse indices are provided on METAX.");
    }
    if (has_tensor(attn_sink) || has_tensor(extra_k_cache)
        || has_tensor(extra_indices_in_kvcache) || has_tensor(topk_length)
        || has_tensor(extra_topk_length)) {
        throw std::runtime_error(
            std::string(op_name)
            + " does not support attn_sink or extended KV-cache inputs on METAX.");
    }
}

double resolve_softmax_scale(const Tensor &q,
                             const std::optional<double> &softmax_scale,
                             const char *op_name) {
    if (softmax_scale.has_value()) {
        return softmax_scale.value();
    }
    if (!q || q->ndim() == 0 || q->size(q->ndim() - 1) == 0) {
        throw std::runtime_error(
            std::string(op_name) + " cannot infer softmax_scale from q.");
    }
    return 1.0 / std::sqrt(static_cast<double>(q->size(q->ndim() - 1)));
}

int checked_int(int64_t value, const char *name, const char *op_name) {
    if (value < static_cast<int64_t>(std::numeric_limits<int>::min())
        || value > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            std::string(op_name) + ": " + name + " is out of int range.");
    }
    return static_cast<int>(value);
}

DataType from_at_scalar_type(at::ScalarType dtype) {
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
    case at::kFloat8_e4m3fn:
        return DataType::F8;
    default:
        throw std::runtime_error(
            "flash_mla_with_kvcache_impl: unsupported FlashMLA return dtype.");
    }
}

Device from_at_device(const at::Device &device) {
    if (device.is_cpu()) {
        return Device(Device::Type::CPU, 0);
    }
    if (!device.is_cuda()) {
        throw std::runtime_error(
            "flash_mla_with_kvcache_impl: unsupported FlashMLA return device.");
    }
    return Device(Device::Type::METAX,
                  static_cast<Device::Index>(device.index()));
}

Shape shape_from_at_tensor(const at::Tensor &tensor) {
    Shape shape;
    shape.reserve(static_cast<size_t>(tensor.dim()));
    for (const auto dim : tensor.sizes()) {
        shape.push_back(static_cast<size_t>(dim));
    }
    return shape;
}

void copy_flashmla_tensor_exact(Tensor &dst,
                                at::Tensor src,
                                const char *name,
                                bool allow_allocate) {
    constexpr const char *op_name = "flash_mla_with_kvcache_impl";
    if (!src.defined()) {
        throw std::runtime_error(
            std::string(op_name) + ": FlashMLA returned undefined " + name + ".");
    }
    src = src.contiguous();
    const auto expected_shape = shape_from_at_tensor(src);
    const auto expected_dtype = from_at_scalar_type(src.scalar_type());
    const auto expected_device = from_at_device(src.device());
    if (!dst) {
        if (!allow_allocate) {
            throw std::runtime_error(
                std::string(op_name) + ": " + name + " must be preallocated.");
        }
        dst = Tensor::empty(expected_shape, expected_dtype, expected_device);
    }
    if (dst->shape() != expected_shape) {
        throw std::runtime_error(
            std::string(op_name) + ": " + name + " shape mismatch.");
    }
    if (dst->dtype() != expected_dtype) {
        throw std::runtime_error(
            std::string(op_name) + ": " + name + " dtype mismatch.");
    }
    if (dst->device() != expected_device) {
        throw std::runtime_error(
            std::string(op_name) + ": " + name + " device mismatch.");
    }
    if (!dst->is_contiguous()) {
        throw std::runtime_error(
            std::string(op_name) + ": " + name + " must be contiguous.");
    }
    auto dst_at = infinicore::adaptor::to_aten_tensor(dst);
    dst_at.copy_(src);
}

void *resolve_flashmla_so_symbol(const char *symbol, const char *op_name) {
    if (void *fn = dlsym(RTLD_DEFAULT, symbol)) {
        return fn;
    }

    const char *so_path = std::getenv("INFINICORE_METAX_FLASH_MLA_SO");
    if (so_path == nullptr || so_path[0] == '\0') {
        so_path = std::getenv("INFINICORE_DSV4_FLASHMLA_SO");
    }
    if (so_path == nullptr || so_path[0] == '\0') {
        so_path = kDefaultFlashMlaSoPath;
    }

    void *handle = dlopen(so_path, RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char *error = dlerror();
        throw std::runtime_error(
            std::string(op_name) + " requires flash_mla_cuda; failed to dlopen "
            + so_path
            + (error == nullptr ? "" : std::string(": ") + error));
    }
    if (void *fn = dlsym(handle, symbol)) {
        return fn;
    }
    throw std::runtime_error(
        std::string(op_name) + " missing flash_mla_cuda symbol: " + symbol);
}

FlashMlaFwdKvcacheMlaFn flashmla_fwd_kvcache_mla_fn(const char *op_name) {
    static auto fn = reinterpret_cast<FlashMlaFwdKvcacheMlaFn>(
        resolve_flashmla_so_symbol(kFlashMlaFwdKvcacheMlaSymbol, op_name));
    return fn;
}

std::optional<const at::Tensor>
to_optional_const_aten(const std::optional<Tensor> &tensor) {
    if (!has_tensor(tensor)) {
        return std::nullopt;
    }
    return infinicore::adaptor::to_aten_tensor(tensor.value());
}

std::optional<graph::GraphTensor>
to_optional_graph_tensor(const std::optional<Tensor> &tensor) {
    if (!has_tensor(tensor)) {
        return std::nullopt;
    }
    return graph::GraphTensor(tensor.value());
}

std::optional<Tensor>
to_optional_tensor(const std::optional<graph::GraphTensor> &tensor) {
    if (!tensor.has_value()) {
        return std::nullopt;
    }
    return tensor.value();
}

void validate_inputs(const Tensor &out,
                     const Tensor &lse,
                     const Tensor &q,
                     const Tensor &k_cache,
                     const Tensor &block_table,
                     const Tensor &cache_seqlens,
                     int64_t head_dim_v,
                     const FlashMLASchedMeta &sched_meta,
                     const std::optional<Tensor> &indices,
                     const char *op_name) {
    check_device(out, op_name);
    check_device(lse, op_name);
    check_device(q, op_name);
    check_device(k_cache, op_name);
    check_device(block_table, op_name);
    check_device(cache_seqlens, op_name);
    check_optional_device(indices, op_name);
    if (!sched_meta.has_valid_sched_meta()) {
        throw std::runtime_error(
            std::string(op_name)
            + " requires valid tile_scheduler_metadata on METAX.");
    }
    check_device(sched_meta.tile_scheduler_metadata, op_name);
    check_device(sched_meta.num_splits, op_name);

    if (q->ndim() != 4 || k_cache->ndim() != 4) {
        throw std::runtime_error(
            std::string(op_name) + " expects four-dimensional q and k_cache.");
    }
    if (q->size(0) == 0 || q->size(1) == 0 || q->size(2) == 0
        || k_cache->size(1) == 0 || k_cache->size(2) == 0) {
        throw std::runtime_error(
            std::string(op_name) + " expects non-zero batch, sequence, and head dimensions.");
    }
    checked_int(head_dim_v, "head_dim_v", op_name);
    if (cache_seqlens->dtype() != DataType::I32
        || block_table->dtype() != DataType::I32
        || cache_seqlens->ndim() != 1
        || cache_seqlens->size(0) != q->size(0)
        || block_table->ndim() != 2
        || block_table->size(0) != q->size(0)) {
        throw std::runtime_error(
            std::string(op_name)
            + " expects int32 cache_seqlens [batch] and block_table [batch, blocks].");
    }
    if (!q->is_contiguous() || !k_cache->is_contiguous()
        || !cache_seqlens->is_contiguous() || !block_table->is_contiguous()) {
        throw std::runtime_error(
            std::string(op_name)
            + " expects contiguous q, k_cache, cache_seqlens, and block_table.");
    }
    if (has_tensor(indices)
        && (indices.value()->dtype() != DataType::I32
            || indices.value()->ndim() != 3
            || indices.value()->size(0) != q->size(0)
            || indices.value()->size(1) != q->size(1)
            || indices.value()->size(2) == 0
            || !indices.value()->is_contiguous())) {
        throw std::runtime_error(
            std::string(op_name)
            + " expects contiguous int32 indices [batch, seq_q, topk].");
    }
    if (q->size(2) > std::numeric_limits<size_t>::max() / q->size(1)
        || (q->size(1) * q->size(2)) % k_cache->size(2) != 0) {
        throw std::runtime_error(
            std::string(op_name)
            + " expects seq_q * query_heads to be divisible by KV heads.");
    }
}

FlashMLASchedMeta make_graph_sched_meta(const FlashMLASchedMeta &metadata) {
    auto graph_metadata = metadata;
    graph_metadata.tile_scheduler_metadata = graph::GraphTensor(metadata.tile_scheduler_metadata);
    graph_metadata.num_splits = graph::GraphTensor(metadata.num_splits);
    return graph_metadata;
}

} // namespace

void fwd_kvcache_mla_impl_internal(
    Tensor &out,
    Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> block_table,
    std::optional<Tensor> cache_seqlens,
    int64_t head_dim_v,
    FlashMLASchedMeta &tile_scheduler_metadata,
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
    check_metax_options(block_table,
                        cache_seqlens,
                        num_splits,
                        causal,
                        indices,
                        attn_sink,
                        extra_k_cache,
                        extra_indices_in_kvcache,
                        topk_length,
                        extra_topk_length,
                        op_name);

    const Tensor &block_table_tensor = block_table.value();
    const Tensor &cache_seqlens_tensor = cache_seqlens.value();
    validate_inputs(out,
                    lse,
                    q,
                    k_cache,
                    block_table_tensor,
                    cache_seqlens_tensor,
                    head_dim_v,
                    tile_scheduler_metadata,
                    indices,
                    op_name);
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());

    auto q_at = infinicore::adaptor::to_aten_tensor(q);
    auto k_cache_at = infinicore::adaptor::to_aten_tensor(k_cache);
    auto cache_seqlens_at = infinicore::adaptor::to_aten_tensor(cache_seqlens_tensor);
    auto block_table_at = infinicore::adaptor::to_aten_tensor(block_table_tensor);
    auto tile_scheduler_metadata_at = infinicore::adaptor::to_aten_tensor(tile_scheduler_metadata.tile_scheduler_metadata);
    auto scheduler_num_splits_at = infinicore::adaptor::to_aten_tensor(tile_scheduler_metadata.num_splits);

    std::optional<const at::Tensor> k_cache_scale_at = std::nullopt;
    std::optional<const at::Tensor> indices_at = to_optional_const_aten(indices);
    std::optional<const at::Tensor> indices_all_valid_per_q_at = std::nullopt;
    std::optional<const at::Tensor> cp_tot_seqlen_k_at = std::nullopt;

    auto flash_out = flashmla_fwd_kvcache_mla_fn(op_name)(q_at,
                                                          k_cache_at,
                                                          k_cache_scale_at,
                                                          checked_int(head_dim_v, "head_dim_v", op_name),
                                                          cache_seqlens_at,
                                                          block_table_at,
                                                          static_cast<float>(resolve_softmax_scale(q, softmax_scale, op_name)),
                                                          causal,
                                                          tile_scheduler_metadata_at,
                                                          scheduler_num_splits_at,
                                                          is_fp8_kvcache,
                                                          indices_at,
                                                          indices_all_valid_per_q_at,
                                                          1,
                                                          0,
                                                          cp_tot_seqlen_k_at);
    if (flash_out.size() != 2) {
        throw std::runtime_error(std::string(op_name) + ": flash_mla_cuda.fwd_kvcache_mla must return two tensors.");
    }
    copy_flashmla_tensor_exact(out, flash_out[0], "out", false);
    copy_flashmla_tensor_exact(lse, flash_out[1], "lse", false);
}

void flash_mla_with_kvcache_impl(
    Tensor &out,
    Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> block_table,
    std::optional<Tensor> cache_seqlens,
    int64_t head_dim_v,
    FlashMLASchedMeta &tile_scheduler_metadata,
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
    fwd_kvcache_mla_impl_internal(out,
                                  lse,
                                  q,
                                  k_cache,
                                  block_table,
                                  cache_seqlens,
                                  head_dim_v,
                                  tile_scheduler_metadata,
                                  num_splits,
                                  softmax_scale,
                                  causal,
                                  is_fp8_kvcache,
                                  indices,
                                  attn_sink,
                                  extra_k_cache,
                                  extra_indices_in_kvcache,
                                  topk_length,
                                  extra_topk_length);
}

namespace {

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor lse;
    graph::GraphTensor q;
    graph::GraphTensor k_cache;
    std::optional<graph::GraphTensor> block_table;
    std::optional<graph::GraphTensor> cache_seqlens;
    int64_t head_dim_v;
    FlashMLASchedMeta tile_scheduler_metadata;
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
           FlashMLASchedMeta &tile_scheduler_metadata,
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
    check_metax_options(block_table,
                        cache_seqlens,
                        num_splits,
                        causal,
                        indices,
                        attn_sink,
                        extra_k_cache,
                        extra_indices_in_kvcache,
                        topk_length,
                        extra_topk_length,
                        "FlashMlaWithKvcache::plan");
    if (!tile_scheduler_metadata.has_valid_sched_meta()) {
        throw std::runtime_error(
            "FlashMlaWithKvcache::plan requires valid scheduler metadata on METAX.");
    }

    return new PlannedMeta{graph::GraphTensor(out),
                           graph::GraphTensor(lse),
                           graph::GraphTensor(q),
                           graph::GraphTensor(k_cache),
                           to_optional_graph_tensor(block_table),
                           to_optional_graph_tensor(cache_seqlens),
                           head_dim_v,
                           make_graph_sched_meta(tile_scheduler_metadata),
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
    fwd_kvcache_mla_impl_internal(
        planned->out,
        planned->lse,
        planned->q,
        planned->k_cache,
        to_optional_tensor(planned->block_table),
        to_optional_tensor(planned->cache_seqlens),
        planned->head_dim_v,
        planned->tile_scheduler_metadata,
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

} // namespace

static bool registered = []() {
    FlashMlaWithKvcache::plan_dispatcher().registerDevice(Device::Type::METAX, &plan);
    FlashMlaWithKvcache::run_dispatcher().registerDevice(Device::Type::METAX, &run);
    FlashMlaWithKvcache::cleanup_dispatcher().registerDevice(Device::Type::METAX, &cleanup);
    flash_mla_with_kvcache_impl_dispatcher().registerDevice(Device::Type::METAX, &flash_mla_with_kvcache_impl);
    return true;
}();

} // namespace infinicore::op::flash_mla::flash_mla_with_kvcache_metax

#endif
