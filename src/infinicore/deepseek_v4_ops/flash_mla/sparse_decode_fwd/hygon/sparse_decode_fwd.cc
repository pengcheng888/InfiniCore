#include "infinicore/ops/flash_mla/sparse_decode_fwd.hpp"

#include "sparse_decode_symbol.hpp"

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

DataType from_at_scalar_type_for_sparse_decode(at::ScalarType dtype) {
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
        throw std::runtime_error("sparse_decode_fwd_impl: unsupported FlashMLA return dtype.");
    }
}

Device from_at_device_for_sparse_decode(const at::Device &device) {
    if (device.is_cpu()) {
        return Device(Device::Type::CPU, 0);
    }
    if (!device.is_cuda()) {
        throw std::runtime_error("sparse_decode_fwd_impl: unsupported FlashMLA return device.");
    }
#if defined(ENABLE_HYGON_API)
    return Device(Device::Type::HYGON, static_cast<Device::Index>(device.index()));
#else
    return Device(Device::Type::NVIDIA, static_cast<Device::Index>(device.index()));
#endif
}

Shape shape_from_at_tensor_for_sparse_decode(const at::Tensor &tensor) {
    Shape shape;
    shape.reserve(static_cast<size_t>(tensor.dim()));
    for (const auto dim : tensor.sizes()) {
        shape.push_back(static_cast<size_t>(dim));
    }
    return shape;
}

void copy_flashmla_return_tensor_exact(Tensor &dst, at::Tensor src, const char *name) {
    if (!src.defined()) {
        throw std::runtime_error(std::string("sparse_decode_fwd_impl: FlashMLA returned undefined ") + name + ".");
    }
    src = src.contiguous();
    const auto expected_shape = shape_from_at_tensor_for_sparse_decode(src);
    const auto expected_dtype = from_at_scalar_type_for_sparse_decode(src.scalar_type());
    const auto expected_device = from_at_device_for_sparse_decode(src.device());
    if (!dst) {
        dst = Tensor::empty(expected_shape, expected_dtype, expected_device);
    }
    if (dst->shape() != expected_shape) {
        throw std::runtime_error(std::string("sparse_decode_fwd_impl: ") + name + " shape mismatch.");
    }
    if (dst->dtype() != expected_dtype) {
        throw std::runtime_error(std::string("sparse_decode_fwd_impl: ") + name + " dtype mismatch.");
    }
    if (dst->device() != expected_device) {
        throw std::runtime_error(std::string("sparse_decode_fwd_impl: ") + name + " device mismatch.");
    }
    if (!dst->is_contiguous()) {
        throw std::runtime_error(std::string("sparse_decode_fwd_impl: ") + name + " must be contiguous.");
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
namespace flash_mla::sparse_decode_fwd_hygon {

void sparse_decode_fwd_impl(
    Tensor &out,
    Tensor &lse,
    Tensor &new_tile_scheduler_metadata_tensor,
    Tensor &new_num_splits_tensor,
    const Tensor &q,
    const Tensor &k_cache,
    const Tensor &indices,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> extra_topk_length,
    int64_t head_dim_v,
    double softmax_scale);

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor lse;
    graph::GraphTensor q;
    graph::GraphTensor k_cache;
    graph::GraphTensor indices;
    std::optional<graph::GraphTensor> topk_length;
    std::optional<graph::GraphTensor> attn_sink;
    std::optional<graph::GraphTensor> extra_k_cache;
    std::optional<graph::GraphTensor> extra_indices_in_kvcache;
    std::optional<graph::GraphTensor> extra_topk_length;
    int64_t head_dim_v;
    double softmax_scale;
};

std::optional<graph::GraphTensor> make_optional_graph_tensor(std::optional<Tensor> tensor) {
    if (!tensor.has_value() || !tensor.value()) {
        return std::nullopt;
    }
    return graph::GraphTensor(tensor.value());
}

void *plan(Tensor out,
           Tensor lse,
           const Tensor &q,
           const Tensor &k_cache,
           const Tensor &indices,
           std::optional<Tensor> topk_length,
           std::optional<Tensor> attn_sink,
           std::optional<Tensor> extra_k_cache,
           std::optional<Tensor> extra_indices_in_kvcache,
           std::optional<Tensor> extra_topk_length,
           int64_t head_dim_v,
           double softmax_scale) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(lse),
        graph::GraphTensor(q),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(indices),
        make_optional_graph_tensor(topk_length),
        make_optional_graph_tensor(attn_sink),
        make_optional_graph_tensor(extra_k_cache),
        make_optional_graph_tensor(extra_indices_in_kvcache),
        make_optional_graph_tensor(extra_topk_length),
        head_dim_v,
        softmax_scale};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    Tensor new_tile_scheduler_metadata;
    Tensor new_num_splits;
    sparse_decode_fwd_impl(planned->out,
                           planned->lse,
                           new_tile_scheduler_metadata,
                           new_num_splits,
                           planned->q,
                           planned->k_cache,
                           planned->indices,
                           planned->topk_length,
                           planned->attn_sink,
                           std::nullopt,
                           std::nullopt,
                           planned->extra_k_cache,
                           planned->extra_indices_in_kvcache,
                           planned->extra_topk_length,
                           planned->head_dim_v,
                           planned->softmax_scale);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::flash_mla::SparseDecodeFwd::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    ::infinicore::op::flash_mla::SparseDecodeFwd::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    ::infinicore::op::flash_mla::SparseDecodeFwd::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

void sparse_decode_fwd_impl(
    Tensor &out,
    Tensor &lse,
    Tensor &new_tile_scheduler_metadata_tensor,
    Tensor &new_num_splits_tensor,
    const Tensor &q,
    const Tensor &k_cache,
    const Tensor &indices,
    std::optional<Tensor> topk_length,
    std::optional<Tensor> attn_sink,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    std::optional<Tensor> extra_k_cache,
    std::optional<Tensor> extra_indices_in_kvcache,
    std::optional<Tensor> extra_topk_length,
    int64_t head_dim_v,
    double softmax_scale) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    constexpr const char *op_name = "sparse_decode_fwd_impl";
    check_device(q, op_name);
    check_device(k_cache, op_name);
    check_device(indices, op_name);
    check_optional_device(tile_scheduler_metadata, op_name);
    check_optional_device(num_splits, op_name);
    check_optional_device(attn_sink, op_name);
    check_optional_device(extra_k_cache, op_name);
    check_optional_device(extra_indices_in_kvcache, op_name);
    check_optional_device(topk_length, op_name);
    check_optional_device(extra_topk_length, op_name);

#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

#if defined(ENABLE_HYGON_API)
    auto q_flash_at = to_aten_tensor_for_flashmla(q);
    auto k_cache_flash_at = to_aten_tensor_for_flashmla(k_cache);
    auto indices_flash_at = to_aten_tensor_for_flashmla(indices);
    auto attn_sink_flash_at = to_optional_aten_for_flashmla(attn_sink);
    auto extra_k_cache_flash_at = to_optional_aten_for_flashmla(extra_k_cache);
    auto extra_indices_flash_at = to_optional_aten_for_flashmla(extra_indices_in_kvcache);
    auto topk_length_flash_at = to_optional_aten_for_flashmla(topk_length);
    auto extra_topk_length_flash_at = to_optional_aten_for_flashmla(extra_topk_length);

    std::optional<at::Tensor> tile_scheduler_metadata_flash_at = to_optional_aten_for_flashmla(tile_scheduler_metadata);
    std::optional<at::Tensor> num_splits_flash_at = to_optional_aten_for_flashmla(num_splits);
    auto [flash_out_at, flash_lse_at, new_tile_scheduler_metadata, new_num_splits] = flashmla_sparse_decode_fn(op_name)(q_flash_at,
                                                                                                                        k_cache_flash_at,
                                                                                                                        indices_flash_at,
                                                                                                                        topk_length_flash_at,
                                                                                                                        attn_sink_flash_at,
                                                                                                                        tile_scheduler_metadata_flash_at,
                                                                                                                        num_splits_flash_at,
                                                                                                                        extra_k_cache_flash_at,
                                                                                                                        extra_indices_flash_at,
                                                                                                                        extra_topk_length_flash_at,
                                                                                                                        static_cast<int>(head_dim_v),
                                                                                                                        static_cast<float>(softmax_scale));
    copy_flashmla_return_tensor_exact(out, flash_out_at, "out");
    copy_flashmla_return_tensor_exact(lse, flash_lse_at, "softmax_lse");
    (void)new_tile_scheduler_metadata_tensor;
    (void)new_num_splits_tensor;
    (void)new_tile_scheduler_metadata;
    (void)new_num_splits;
    return;
#endif

    throw std::runtime_error("sparse_decode_fwd_impl only supports HYGON FlashMLA sparse decode.");
#endif
    (void)out;
    (void)lse;
    (void)new_tile_scheduler_metadata_tensor;
    (void)new_num_splits_tensor;
    (void)q;
    (void)k_cache;
    (void)indices;
    (void)topk_length;
    (void)attn_sink;
    (void)tile_scheduler_metadata;
    (void)num_splits;
    (void)extra_k_cache;
    (void)extra_indices_in_kvcache;
    (void)extra_topk_length;
    (void)head_dim_v;
    (void)softmax_scale;
    throw std::runtime_error("sparse_decode_fwd_impl requires an ATen-enabled HYGON/NVIDIA build.");
}

} // namespace flash_mla::sparse_decode_fwd_hygon
#endif

} // namespace infinicore::op
