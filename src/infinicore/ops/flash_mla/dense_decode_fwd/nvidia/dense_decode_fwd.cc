#include "infinicore/ops/flash_mla/dense_decode_fwd.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include <cstdlib>
#include <dlfcn.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

namespace infinicore::op {

#if defined(ENABLE_ATEN) && defined(ENABLE_NVIDIA_API)
namespace {

constexpr const char *kFlashMlaOpName = "_flashmla_C::dense_decode_fwd";
constexpr const char *kDefaultFlashMlaSoPath = "/usr/local/lib/python3.12/dist-packages/vllm/_flashmla_C.abi3.so";

using FlashMlaDenseDecodeSchema = std::tuple<at::Tensor,
                                             at::Tensor,
                                             std::optional<at::Tensor>,
                                             std::optional<at::Tensor>>(
    at::Tensor,
    at::Tensor,
    int64_t,
    at::Tensor,
    at::Tensor,
    double,
    bool,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>);

void ensure_flashmla_registered(const char *op_name) {
    static bool loaded = []() {
        const char *so_path = std::getenv("INFINICORE_NVIDIA_FLASHMLA_SO");
        if (so_path == nullptr || so_path[0] == '\0') {
            so_path = std::getenv("INFINICORE_DSV4_FLASHMLA_SO");
        }
        if (so_path == nullptr || so_path[0] == '\0') {
            so_path = kDefaultFlashMlaSoPath;
        }

        void *handle = dlopen(so_path, RTLD_NOW | RTLD_GLOBAL);
        if (handle == nullptr) {
            const char *err = dlerror();
            throw std::runtime_error(std::string("dense_decode_fwd_impl requires vllm._flashmla_C; failed to dlopen ")
                                     + so_path
                                     + (err == nullptr ? "" : std::string(": ") + err));
        }
        return true;
    }();
    (void)loaded;
    (void)op_name;
}

auto &flashmla_dense_decode_op() {
    ensure_flashmla_registered("dense_decode_fwd_impl");
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow(kFlashMlaOpName, "")
                         .typed<FlashMlaDenseDecodeSchema>();
    return op;
}

void check_device(const Tensor &tensor, const char *op_name) {
    if (tensor->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error(std::string(op_name) + " expects NVIDIA tensors in this build.");
    }
}

void check_optional_device(const std::optional<Tensor> &tensor, const char *op_name) {
    if (tensor.has_value() && tensor.value()) {
        check_device(*tensor, op_name);
    }
}

std::optional<at::Tensor> to_optional_aten_tensor(const std::optional<Tensor> &tensor) {
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
    case at::kFloat8_e4m3fn:
        return DataType::F8;
    default:
        throw std::runtime_error("dense_decode_fwd_impl: unsupported FlashMLA return dtype.");
    }
}

Device from_at_device_for_dense_decode(const at::Device &device) {
    if (device.is_cpu()) {
        return Device(Device::Type::CPU, 0);
    }
    if (!device.is_cuda()) {
        throw std::runtime_error("dense_decode_fwd_impl: unsupported FlashMLA return device.");
    }
    return Device(Device::Type::NVIDIA, static_cast<Device::Index>(device.index()));
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
        throw std::runtime_error(std::string("dense_decode_fwd_impl: FlashMLA returned undefined ") + name + ".");
    }
    src = src.contiguous();
    const auto expected_shape = shape_from_at_tensor_for_dense_decode(src);
    const auto expected_dtype = from_at_scalar_type_for_dense_decode(src.scalar_type());
    const auto expected_device = from_at_device_for_dense_decode(src.device());
    if (!dst) {
        dst = infinicore::adaptor::from_aten_tensor(src);
        return;
    }
    if (dst->shape() != expected_shape) {
        throw std::runtime_error(std::string("dense_decode_fwd_impl: ") + name + " shape mismatch.");
    }
    if (dst->dtype() != expected_dtype) {
        throw std::runtime_error(std::string("dense_decode_fwd_impl: ") + name + " dtype mismatch.");
    }
    if (dst->device() != expected_device) {
        throw std::runtime_error(std::string("dense_decode_fwd_impl: ") + name + " device mismatch.");
    }
    if (!dst->is_contiguous()) {
        throw std::runtime_error(std::string("dense_decode_fwd_impl: ") + name + " must be contiguous.");
    }
    auto dst_at = infinicore::adaptor::to_aten_tensor(dst);
    dst_at.copy_(src);
}

} // namespace

namespace flash_mla::dense_decode_fwd_nvidia {

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor lse;
    std::optional<graph::GraphTensor> new_tile_scheduler_metadata;
    std::optional<graph::GraphTensor> new_num_splits;
    graph::GraphTensor q;
    graph::GraphTensor k_cache;
    int64_t head_dim_v;
    graph::GraphTensor cache_seqlens;
    graph::GraphTensor block_table;
    double softmax_scale;
    bool causal;
    std::optional<graph::GraphTensor> tile_scheduler_metadata;
    std::optional<graph::GraphTensor> num_splits;
};

void dense_decode_fwd_impl(
    Tensor &out,
    Tensor &lse,
    std::optional<Tensor> &new_tile_scheduler_metadata,
    std::optional<Tensor> &new_num_splits,
    const Tensor &q,
    const Tensor &k_cache,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits);

void *plan(Tensor out,
           Tensor lse,
           std::optional<Tensor> new_tile_scheduler_metadata,
           std::optional<Tensor> new_num_splits,
           const Tensor &q,
           const Tensor &k_cache,
           int64_t head_dim_v,
           const Tensor &cache_seqlens,
           const Tensor &block_table,
           double softmax_scale,
           bool causal,
           std::optional<Tensor> tile_scheduler_metadata,
           std::optional<Tensor> num_splits) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(lse),
        to_optional_graph_tensor(new_tile_scheduler_metadata),
        to_optional_graph_tensor(new_num_splits),
        graph::GraphTensor(q),
        graph::GraphTensor(k_cache),
        head_dim_v,
        graph::GraphTensor(cache_seqlens),
        graph::GraphTensor(block_table),
        softmax_scale,
        causal,
        to_optional_graph_tensor(tile_scheduler_metadata),
        to_optional_graph_tensor(num_splits)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    std::optional<Tensor> new_tile_scheduler_metadata = to_optional_tensor(planned->new_tile_scheduler_metadata);
    std::optional<Tensor> new_num_splits = to_optional_tensor(planned->new_num_splits);
    std::optional<Tensor> tile_scheduler_metadata = to_optional_tensor(planned->tile_scheduler_metadata);
    std::optional<Tensor> num_splits = to_optional_tensor(planned->num_splits);

    dense_decode_fwd_impl(planned->out,
                          planned->lse,
                          new_tile_scheduler_metadata,
                          new_num_splits,
                          planned->q,
                          planned->k_cache,
                          planned->head_dim_v,
                          planned->cache_seqlens,
                          planned->block_table,
                          planned->softmax_scale,
                          planned->causal,
                          tile_scheduler_metadata,
                          num_splits);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::flash_mla::DenseDecodeFwd::plan_dispatcher().registerDevice(Device::Type::NVIDIA, &plan);
    ::infinicore::op::flash_mla::DenseDecodeFwd::run_dispatcher().registerDevice(Device::Type::NVIDIA, &run);
    ::infinicore::op::flash_mla::DenseDecodeFwd::cleanup_dispatcher().registerDevice(Device::Type::NVIDIA, &cleanup);
    ::infinicore::op::flash_mla::dense_decode_fwd_impl_dispatcher().registerDevice(Device::Type::NVIDIA, &dense_decode_fwd_impl);
    return true;
}();

void dense_decode_fwd_impl(
    Tensor &out,
    Tensor &lse,
    std::optional<Tensor> &new_tile_scheduler_metadata_out,
    std::optional<Tensor> &new_num_splits_out,
    const Tensor &q,
    const Tensor &k_cache,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits) {
    constexpr const char *op_name = "dense_decode_fwd_impl";
    check_device(q, op_name);
    check_device(k_cache, op_name);
    check_device(cache_seqlens, op_name);
    check_device(block_table, op_name);
    check_optional_device(new_tile_scheduler_metadata_out, op_name);
    check_optional_device(new_num_splits_out, op_name);
    check_optional_device(tile_scheduler_metadata, op_name);
    check_optional_device(num_splits, op_name);

    if (k_cache->ndim() != 4) {
        throw std::runtime_error("dense_decode_fwd_impl: NVIDIA FlashMLA expects k_cache shape [blocks, page_size, kv_heads, head_dim].");
    }
    if (cache_seqlens->dtype() != DataType::I32 || block_table->dtype() != DataType::I32) {
        throw std::runtime_error("dense_decode_fwd_impl: NVIDIA FlashMLA expects int32 cache_seqlens and block_table.");
    }
    if (!q->is_contiguous() || !k_cache->is_contiguous() || !cache_seqlens->is_contiguous() || !block_table->is_contiguous()) {
        throw std::runtime_error("dense_decode_fwd_impl: NVIDIA FlashMLA expects contiguous q/k_cache/cache_seqlens/block_table.");
    }

    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());

    auto q_at = infinicore::adaptor::to_aten_tensor(q);
    auto k_cache_at = infinicore::adaptor::to_aten_tensor(k_cache);
    auto cache_seqlens_at = infinicore::adaptor::to_aten_tensor(cache_seqlens);
    auto block_table_at = infinicore::adaptor::to_aten_tensor(block_table);
    auto tile_scheduler_metadata_at = to_optional_aten_tensor(tile_scheduler_metadata);
    auto num_splits_at = to_optional_aten_tensor(num_splits);
    std::optional<at::Tensor> out_at = infinicore::adaptor::to_aten_tensor(out);

    auto [flash_out_at, flash_lse_at, new_tile_scheduler_metadata, new_num_splits] =
        flashmla_dense_decode_op().call(q_at,
                                        k_cache_at,
                                        head_dim_v,
                                        cache_seqlens_at,
                                        block_table_at,
                                        softmax_scale,
                                        causal,
                                        tile_scheduler_metadata_at,
                                        num_splits_at,
                                        out_at);

    copy_flashmla_return_tensor_exact(out, flash_out_at, "out");
    copy_flashmla_return_tensor_exact(lse, flash_lse_at, "softmax_lse");

    if (!new_tile_scheduler_metadata.has_value() || !new_num_splits.has_value()) {
        throw std::runtime_error("dense_decode_fwd_impl: FlashMLA returned None scheduler metadata.");
    }
    if (!new_tile_scheduler_metadata_out.has_value() || !new_tile_scheduler_metadata_out.value()) {
        at::Tensor src = new_tile_scheduler_metadata.value().contiguous();
        new_tile_scheduler_metadata_out = Tensor::empty(shape_from_at_tensor_for_dense_decode(src),
                                                        from_at_scalar_type_for_dense_decode(src.scalar_type()),
                                                        from_at_device_for_dense_decode(src.device()));
    }
    if (!new_num_splits_out.has_value() || !new_num_splits_out.value()) {
        at::Tensor src = new_num_splits.value().contiguous();
        new_num_splits_out = Tensor::empty(shape_from_at_tensor_for_dense_decode(src),
                                           from_at_scalar_type_for_dense_decode(src.scalar_type()),
                                           from_at_device_for_dense_decode(src.device()));
    }
    copy_flashmla_return_tensor_exact(new_tile_scheduler_metadata_out.value(),
                                      new_tile_scheduler_metadata.value(),
                                      "new_tile_scheduler_metadata");
    copy_flashmla_return_tensor_exact(new_num_splits_out.value(),
                                      new_num_splits.value(),
                                      "new_num_splits");
}

} // namespace flash_mla::dense_decode_fwd_nvidia
#endif

} // namespace infinicore::op
