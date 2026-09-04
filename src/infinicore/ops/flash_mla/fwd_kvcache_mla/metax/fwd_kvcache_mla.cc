#include "fwd_kvcache_mla.hpp"

#if defined(ENABLE_ATEN) && defined(ENABLE_METAX_API)

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op::flash_mla::fwd_kvcache_mla_metax {
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

void check_device(const Tensor &tensor, const char *op_name) {
    if (!tensor || tensor->device().getType() != Device::Type::METAX) {
        throw std::runtime_error(std::string(op_name) + " expects METAX tensors.");
    }
}

void check_optional_device(const std::optional<Tensor> &tensor, const char *op_name) {
    if (tensor.has_value() && tensor.value()) {
        check_device(*tensor, op_name);
    }
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
        throw std::runtime_error("fwd_kvcache_mla_impl: unsupported FlashMLA return dtype.");
    }
}

Device from_at_device(const at::Device &device) {
    if (device.is_cpu()) {
        return Device(Device::Type::CPU, 0);
    }
    if (!device.is_cuda()) {
        throw std::runtime_error("fwd_kvcache_mla_impl: unsupported FlashMLA return device.");
    }
    return Device(Device::Type::METAX, static_cast<Device::Index>(device.index()));
}

Shape shape_from_at_tensor(const at::Tensor &tensor) {
    Shape shape;
    shape.reserve(static_cast<size_t>(tensor.dim()));
    for (const auto dim : tensor.sizes()) {
        shape.push_back(static_cast<size_t>(dim));
    }
    return shape;
}

void copy_flashmla_return_tensor_exact(Tensor &dst, at::Tensor src, const char *name) {
    if (!src.defined()) {
        throw std::runtime_error(std::string("fwd_kvcache_mla_impl: FlashMLA returned undefined ") + name + ".");
    }
    src = src.contiguous();
    const auto expected_shape = shape_from_at_tensor(src);
    const auto expected_dtype = from_at_scalar_type(src.scalar_type());
    const auto expected_device = from_at_device(src.device());
    if (!dst) {
        dst = Tensor::empty(expected_shape, expected_dtype, expected_device);
    }
    if (dst->shape() != expected_shape) {
        throw std::runtime_error(std::string("fwd_kvcache_mla_impl: ") + name + " shape mismatch.");
    }
    if (dst->dtype() != expected_dtype) {
        throw std::runtime_error(std::string("fwd_kvcache_mla_impl: ") + name + " dtype mismatch.");
    }
    if (dst->device() != expected_device) {
        throw std::runtime_error(std::string("fwd_kvcache_mla_impl: ") + name + " device mismatch.");
    }
    if (!dst->is_contiguous()) {
        throw std::runtime_error(std::string("fwd_kvcache_mla_impl: ") + name + " must be contiguous.");
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
        const char *err = dlerror();
        throw std::runtime_error(std::string(op_name) + " requires flash_mla_cuda; failed to dlopen " + so_path + (err == nullptr ? "" : std::string(": ") + err));
    }
    if (void *fn = dlsym(handle, symbol)) {
        return fn;
    }
    throw std::runtime_error(std::string(op_name) + " missing flash_mla_cuda symbol: " + symbol);
}

FlashMlaFwdKvcacheMlaFn flashmla_fwd_kvcache_mla_fn(const char *op_name) {
    static auto fn = reinterpret_cast<FlashMlaFwdKvcacheMlaFn>(
        resolve_flashmla_so_symbol(kFlashMlaFwdKvcacheMlaSymbol, op_name));
    return fn;
}

std::optional<const at::Tensor> to_optional_const_aten(const std::optional<Tensor> &tensor) {
    if (!tensor.has_value() || !tensor.value()) {
        return std::nullopt;
    }
    return infinicore::adaptor::to_aten_tensor(tensor.value());
}

std::optional<graph::GraphTensor> to_optional_graph_tensor(const std::optional<Tensor> &tensor) {
    if (!tensor.has_value() || !tensor.value()) {
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

void fwd_kvcache_mla_impl(
    Tensor &out,
    Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    std::optional<Tensor> k_cache_scale,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    const Tensor &tile_scheduler_metadata,
    const Tensor &num_splits,
    bool is_fp8_kvcache,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_block_table,
    int64_t cp_world_size,
    int64_t cp_rank,
    std::optional<Tensor> cp_tot_seqused_k) {
    constexpr const char *op_name = "fwd_kvcache_mla_impl";
    check_device(q, op_name);
    check_device(k_cache, op_name);
    check_device(cache_seqlens, op_name);
    check_device(block_table, op_name);
    check_device(tile_scheduler_metadata, op_name);
    check_device(num_splits, op_name);
    check_optional_device(k_cache_scale, op_name);
    check_optional_device(extra_k_cache, op_name);
    check_optional_device(extra_block_table, op_name);
    check_optional_device(cp_tot_seqused_k, op_name);

    if (head_dim_v <= 0 || cp_world_size <= 0 || cp_rank < 0 || cp_rank >= cp_world_size) {
        throw std::runtime_error("fwd_kvcache_mla_impl received invalid scalar parameters.");
    }

    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());

    auto q_at = infinicore::adaptor::to_aten_tensor(q);
    auto k_cache_at = infinicore::adaptor::to_aten_tensor(k_cache);
    auto cache_seqlens_at = infinicore::adaptor::to_aten_tensor(cache_seqlens);
    auto block_table_at = infinicore::adaptor::to_aten_tensor(block_table);
    auto tile_scheduler_metadata_at = infinicore::adaptor::to_aten_tensor(tile_scheduler_metadata);
    auto num_splits_at = infinicore::adaptor::to_aten_tensor(num_splits);

    std::optional<const at::Tensor> k_cache_scale_at = to_optional_const_aten(k_cache_scale);
    std::optional<const at::Tensor> extra_k_cache_at = to_optional_const_aten(extra_k_cache);
    std::optional<const at::Tensor> extra_block_table_at = to_optional_const_aten(extra_block_table);
    std::optional<const at::Tensor> cp_tot_seqused_k_at = to_optional_const_aten(cp_tot_seqused_k);

    auto flash_out = flashmla_fwd_kvcache_mla_fn(op_name)(q_at,
                                                          k_cache_at,
                                                          k_cache_scale_at,
                                                          static_cast<int>(head_dim_v),
                                                          cache_seqlens_at,
                                                          block_table_at,
                                                          static_cast<float>(softmax_scale),
                                                          causal,
                                                          tile_scheduler_metadata_at,
                                                          num_splits_at,
                                                          is_fp8_kvcache,
                                                          extra_k_cache_at,
                                                          extra_block_table_at,
                                                          static_cast<int>(cp_world_size),
                                                          static_cast<int>(cp_rank),
                                                          cp_tot_seqused_k_at);
    if (flash_out.size() != 2) {
        throw std::runtime_error("fwd_kvcache_mla_impl: flash_mla_cuda.fwd_kvcache_mla must return two tensors.");
    }
    copy_flashmla_return_tensor_exact(out, flash_out[0], "out");
    copy_flashmla_return_tensor_exact(lse, flash_out[1], "softmax_lse");
}

namespace {

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor lse;
    graph::GraphTensor q;
    graph::GraphTensor k_cache;
    std::optional<graph::GraphTensor> k_cache_scale;
    int64_t head_dim_v;
    graph::GraphTensor cache_seqlens;
    graph::GraphTensor block_table;
    double softmax_scale;
    bool causal;
    graph::GraphTensor tile_scheduler_metadata;
    graph::GraphTensor num_splits;
    bool is_fp8_kvcache;
    std::optional<graph::GraphTensor> extra_k_cache;
    std::optional<graph::GraphTensor> extra_block_table;
    int64_t cp_world_size;
    int64_t cp_rank;
    std::optional<graph::GraphTensor> cp_tot_seqused_k;
};

void *plan(Tensor out,
           Tensor lse,
           const Tensor &q,
           const Tensor &k_cache,
           std::optional<Tensor> k_cache_scale,
           int64_t head_dim_v,
           const Tensor &cache_seqlens,
           const Tensor &block_table,
           double softmax_scale,
           bool causal,
           const Tensor &tile_scheduler_metadata,
           const Tensor &num_splits,
           bool is_fp8_kvcache,
           std::optional<Tensor> extra_k_cache,
           std::optional<Tensor> extra_block_table,
           int64_t cp_world_size,
           int64_t cp_rank,
           std::optional<Tensor> cp_tot_seqused_k) {
    return new PlannedMeta{graph::GraphTensor(out),
                           graph::GraphTensor(lse),
                           graph::GraphTensor(q),
                           graph::GraphTensor(k_cache),
                           to_optional_graph_tensor(k_cache_scale),
                           head_dim_v,
                           graph::GraphTensor(cache_seqlens),
                           graph::GraphTensor(block_table),
                           softmax_scale,
                           causal,
                           graph::GraphTensor(tile_scheduler_metadata),
                           graph::GraphTensor(num_splits),
                           is_fp8_kvcache,
                           to_optional_graph_tensor(extra_k_cache),
                           to_optional_graph_tensor(extra_block_table),
                           cp_world_size,
                           cp_rank,
                           to_optional_graph_tensor(cp_tot_seqused_k)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    fwd_kvcache_mla_impl(planned->out,
                         planned->lse,
                         planned->q,
                         planned->k_cache,
                         to_optional_tensor(planned->k_cache_scale),
                         planned->head_dim_v,
                         planned->cache_seqlens,
                         planned->block_table,
                         planned->softmax_scale,
                         planned->causal,
                         planned->tile_scheduler_metadata,
                         planned->num_splits,
                         planned->is_fp8_kvcache,
                         to_optional_tensor(planned->extra_k_cache),
                         to_optional_tensor(planned->extra_block_table),
                         planned->cp_world_size,
                         planned->cp_rank,
                         to_optional_tensor(planned->cp_tot_seqused_k));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace

static bool registered = []() {
    ::infinicore::op::flash_mla::FwdKvcacheMla::plan_dispatcher().registerDevice(Device::Type::METAX, &plan);
    ::infinicore::op::flash_mla::FwdKvcacheMla::run_dispatcher().registerDevice(Device::Type::METAX, &run);
    ::infinicore::op::flash_mla::FwdKvcacheMla::cleanup_dispatcher().registerDevice(Device::Type::METAX, &cleanup);
    ::infinicore::op::flash_mla::fwd_kvcache_mla_impl_dispatcher().registerDevice(Device::Type::METAX, &fwd_kvcache_mla_impl);
    return true;
}();

} // namespace infinicore::op::flash_mla::fwd_kvcache_mla_metax

#endif
