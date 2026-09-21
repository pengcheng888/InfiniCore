#pragma once

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/ops/flash_mla/flash_mla_with_kvcache.hpp"

#include <ATen/ATen.h>

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op::flash_mla::flash_mla_with_kvcache_hygon::detail {

inline void check_device(const Tensor &tensor, const char *op_name) {
    if (!tensor || tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors.");
    }
}

inline void check_optional_device(const std::optional<Tensor> &tensor, const char *op_name) {
    if (tensor.has_value() && tensor.value()) {
        check_device(*tensor, op_name);
    }
}

inline bool has_tensor(const std::optional<Tensor> &tensor) {
    return tensor.has_value() && tensor.value();
}

inline void check_hygon_dense_decode_options(const std::optional<Tensor> &block_table,
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
    if (has_tensor(indices) || has_tensor(attn_sink) || has_tensor(extra_k_cache)
        || has_tensor(extra_indices_in_kvcache) || has_tensor(topk_length) || has_tensor(extra_topk_length)) {
        throw std::runtime_error(std::string(op_name) + " currently supports dense attention only on HYGON.");
    }
}

inline bool use_sparse_decode(const std::optional<Tensor> &indices) {
    return has_tensor(indices);
}

inline void check_hygon_sparse_decode_options(const std::optional<Tensor> &block_table,
                                              const std::optional<Tensor> &cache_seqlens,
                                              const std::optional<Tensor> &num_splits,
                                              bool is_fp8_kvcache,
                                              const std::optional<Tensor> &indices,
                                              const std::optional<Tensor> &topk_length,
                                              const char *op_name) {
    if (has_tensor(block_table) || has_tensor(cache_seqlens)) {
        throw std::runtime_error(std::string(op_name) + " does not take block_table/cache_seqlens on HYGON sparse decode.");
    }
    if (has_tensor(num_splits)) {
        throw std::runtime_error(std::string(op_name) + " does not support the num_splits override on HYGON sparse decode.");
    }
    if (!is_fp8_kvcache) {
        throw std::runtime_error(std::string(op_name) + " requires is_fp8_kvcache=true on HYGON sparse decode.");
    }
    if (!has_tensor(indices) || !has_tensor(topk_length)) {
        throw std::runtime_error(std::string(op_name) + " requires indices and topk_length on HYGON sparse decode.");
    }
}

inline void check_hygon_decode_options(const std::optional<Tensor> &block_table,
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
    if (use_sparse_decode(indices)) {
        check_hygon_sparse_decode_options(block_table,
                                          cache_seqlens,
                                          num_splits,
                                          is_fp8_kvcache,
                                          indices,
                                          topk_length,
                                          op_name);
        return;
    }
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
}

inline double resolve_softmax_scale(const Tensor &q,
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

inline DataType from_at_scalar_type_for_dense_decode(at::ScalarType dtype) {
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

inline Device from_at_device_for_dense_decode(const at::Device &device) {
    if (device.is_cpu()) {
        return Device(Device::Type::CPU, 0);
    }
    if (!device.is_cuda()) {
        throw std::runtime_error("flash_mla_with_kvcache_impl: unsupported FlashMLA return device.");
    }
    return Device(Device::Type::HYGON, static_cast<Device::Index>(device.index()));
}

inline Shape shape_from_at_tensor_for_dense_decode(const at::Tensor &tensor) {
    Shape shape;
    shape.reserve(static_cast<size_t>(tensor.dim()));
    for (const auto dim : tensor.sizes()) {
        shape.push_back(static_cast<size_t>(dim));
    }
    return shape;
}

inline Tensor empty_like(const at::Tensor &src,
                         std::optional<DataType> dtype = std::nullopt,
                         std::optional<Device> device = std::nullopt) {
    return Tensor::empty(shape_from_at_tensor_for_dense_decode(src),
                         dtype.has_value() ? dtype.value() : from_at_scalar_type_for_dense_decode(src.scalar_type()),
                         device.has_value() ? device.value() : from_at_device_for_dense_decode(src.device()));
}

inline void copy_flashmla_tensor_exact(Tensor &dst, at::Tensor src, const char *name) {
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

inline bool can_reuse_as_output_buffer(const Tensor &dst,
                                      const at::Tensor &src,
                                      DataType dtype,
                                      const Device &device) {
    return dst && dst->shape() == shape_from_at_tensor_for_dense_decode(src)
        && dst->dtype() == dtype
        && dst->device() == device
        && dst->is_contiguous();
}

inline at::Tensor to_aten_tensor_for_flashmla(const Tensor &tensor) {
    if (tensor->dtype() == DataType::F8) {
        std::vector<int64_t> sizes(tensor->shape().begin(), tensor->shape().end());
        std::vector<int64_t> strides(tensor->strides().begin(), tensor->strides().end());
        auto options = at::TensorOptions().dtype(at::ScalarType::Float8_e4m3fn).device(infinicore::adaptor::to_at_device(tensor->device())).requires_grad(false);
        auto *data = const_cast<std::byte *>(tensor->data());
        return at::from_blob(data, sizes, strides, [](void *) {}, options);
    }
    return infinicore::adaptor::to_aten_tensor(tensor);
}

inline std::optional<graph::GraphTensor> to_optional_graph_tensor(const std::optional<Tensor> &tensor) {
    if (!has_tensor(tensor)) {
        return std::nullopt;
    }
    return graph::GraphTensor(tensor.value());
}

inline std::optional<Tensor> to_optional_tensor(const std::optional<graph::GraphTensor> &tensor) {
    if (!tensor.has_value()) {
        return std::nullopt;
    }
    return tensor.value();
}

inline std::optional<at::Tensor> to_optional_aten_for_flashmla(const std::optional<Tensor> &tensor) {
    if (!has_tensor(tensor)) {
        return std::nullopt;
    }
    return to_aten_tensor_for_flashmla(tensor.value());
}

} // namespace infinicore::op::flash_mla::flash_mla_with_kvcache_hygon::detail

#endif
