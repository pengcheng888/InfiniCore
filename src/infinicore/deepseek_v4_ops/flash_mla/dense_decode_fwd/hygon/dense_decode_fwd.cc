#include "infinicore/ops/flash_mla/dense_decode_fwd.hpp"

#include "dense_decode_symbol.hpp"
#include "infinicore/context/context.hpp"

#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace infinicore::op {

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
namespace {

void check_device(const Tensor &tensor, const char *op_name) {
#if defined(ENABLE_HYGON_API)
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
#else
    if (tensor->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error(std::string(op_name) + " expects NVIDIA tensors in this build.");
    }
#endif
}

void check_optional_device(const std::optional<Tensor> &tensor, const char *op_name) {
    if (tensor.has_value() && tensor.value()) {
        check_device(*tensor, op_name);
    }
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
#if defined(ENABLE_HYGON_API)
    case at::kFloat8_e4m3fnuz:
        return DataType::F8;
#else
    case at::kFloat8_e4m3fn:
        return DataType::F8;
#endif
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
#if defined(ENABLE_HYGON_API)
    return Device(Device::Type::HYGON, static_cast<Device::Index>(device.index()));
#else
    return Device(Device::Type::NVIDIA, static_cast<Device::Index>(device.index()));
#endif
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

#if defined(ENABLE_HYGON_API)

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

std::optional<at::Tensor> to_optional_aten_for_flashmla(const std::optional<Tensor> &tensor) {
    if (!tensor.has_value() || !tensor.value()) {
        return std::nullopt;
    }
    return to_aten_tensor_for_flashmla(*tensor);
}

#endif

} // namespace
#endif

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
namespace flash_mla::dense_decode_fwd_hygon {

std::tuple<Tensor, Tensor> dense_decode_fwd_impl(
    Tensor &out,
    Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits);

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor lse;
    graph::GraphTensor q;
    graph::GraphTensor k_cache;
    int64_t head_dim_v;
    graph::GraphTensor cache_seqlens;
    graph::GraphTensor block_table;
    double softmax_scale;
    bool causal;
    graph::GraphTensor tile_scheduler_metadata;
    graph::GraphTensor num_splits;
};

void *plan(Tensor out,
           Tensor lse,
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
        graph::GraphTensor(q),
        graph::GraphTensor(k_cache),
        head_dim_v,
        graph::GraphTensor(cache_seqlens),
        graph::GraphTensor(block_table),
        softmax_scale,
        causal,
        graph::GraphTensor(tile_scheduler_metadata.value()),
        graph::GraphTensor(num_splits.value())};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    (void)dense_decode_fwd_impl(planned->out,
                                planned->lse,
                                planned->q,
                                planned->k_cache,
                                planned->head_dim_v,
                                planned->cache_seqlens,
                                planned->block_table,
                                planned->softmax_scale,
                                planned->causal,
                                std::optional<Tensor>(planned->tile_scheduler_metadata),
                                std::optional<Tensor>(planned->num_splits));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::flash_mla::DenseDecodeFwd::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    ::infinicore::op::flash_mla::DenseDecodeFwd::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    ::infinicore::op::flash_mla::DenseDecodeFwd::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    ::infinicore::op::flash_mla::dense_decode_fwd_impl_dispatcher().registerDevice(Device::Type::HYGON, &dense_decode_fwd_impl);
    return true;
}();

std::tuple<Tensor, Tensor> dense_decode_fwd_impl(
    Tensor &out,
    Tensor &lse,
    const Tensor &q,
    const Tensor &k_cache,
    int64_t head_dim_v,
    const Tensor &cache_seqlens,
    const Tensor &block_table,
    double softmax_scale,
    bool causal,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    constexpr const char *op_name = "dense_decode_fwd_impl";
    check_device(q, op_name);
    check_device(k_cache, op_name);
    check_device(cache_seqlens, op_name);
    check_device(block_table, op_name);
    check_optional_device(tile_scheduler_metadata, op_name);
    check_optional_device(num_splits, op_name);

#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

#if defined(ENABLE_HYGON_API)
    auto q_flash_at = to_aten_tensor_for_flashmla(q);
    auto k_cache_flash_at = to_aten_tensor_for_flashmla(k_cache);
    auto cache_seqlens_flash_at = to_aten_tensor_for_flashmla(cache_seqlens);
    auto block_table_flash_at = to_aten_tensor_for_flashmla(block_table);
    std::optional<at::Tensor> tile_scheduler_metadata_flash_at = to_optional_aten_for_flashmla(tile_scheduler_metadata);
    std::optional<at::Tensor> num_splits_flash_at = to_optional_aten_for_flashmla(num_splits);
    auto [flash_out_at, flash_lse_at, new_tile_scheduler_metadata, new_num_splits] = flashmla_dense_decode_fn(op_name)(q_flash_at,
                                                                                                                       k_cache_flash_at,
                                                                                                                       static_cast<int>(head_dim_v),
                                                                                                                       cache_seqlens_flash_at,
                                                                                                                       block_table_flash_at,
                                                                                                                       static_cast<float>(softmax_scale),
                                                                                                                       causal,
                                                                                                                       tile_scheduler_metadata_flash_at,
                                                                                                                       num_splits_flash_at);
    copy_flashmla_return_tensor_exact(out, flash_out_at, "out");
    copy_flashmla_return_tensor_exact(lse, flash_lse_at, "softmax_lse");
    if (!new_tile_scheduler_metadata.has_value() || !new_num_splits.has_value()) {
        throw std::runtime_error("dense_decode_fwd_impl: FlashMLA returned None scheduler metadata.");
    }
    return {infinicore::adaptor::from_aten_tensor(new_tile_scheduler_metadata.value().contiguous()),
            infinicore::adaptor::from_aten_tensor(new_num_splits.value().contiguous())};
#endif

    throw std::runtime_error("dense_decode_fwd_impl only supports HYGON FlashMLA dense decode.");
#endif
    (void)out;
    (void)lse;
    (void)q;
    (void)k_cache;
    (void)head_dim_v;
    (void)cache_seqlens;
    (void)block_table;
    (void)softmax_scale;
    (void)causal;
    (void)tile_scheduler_metadata;
    (void)num_splits;
    throw std::runtime_error("dense_decode_fwd_impl requires an ATen-enabled HYGON/NVIDIA build.");
}

} // namespace flash_mla::dense_decode_fwd_hygon
#endif

} // namespace infinicore::op
