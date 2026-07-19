#include "infinicore/ops/qwen3_store_kvcache.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <stdexcept>

namespace infinicore::op {

namespace {

size_t row_size(const Tensor &x) {
    if (x->ndim() < 2) {
        throw std::runtime_error("qwen3_store_kvcache_ expects source tensors with rank >= 2.");
    }
    size_t row = 1;
    const auto &shape = x->shape();
    for (size_t i = 1; i < shape.size(); ++i) {
        row *= shape[i];
    }
    return row;
}

} // namespace

void qwen3_store_kvcache_(const Tensor &k,
                          const Tensor &v,
                          Tensor k_cache,
                          Tensor v_cache,
                          const Tensor &indices) {
#if defined(ENABLE_ATEN) && defined(ENABLE_NVIDIA_API)
    if (k->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("qwen3_store_kvcache_ currently supports NVIDIA tensors only.");
    }
    if (k->shape() != v->shape()) {
        throw std::runtime_error("qwen3_store_kvcache_ expects k and v to have the same shape.");
    }
    if (k_cache->shape() != v_cache->shape()) {
        throw std::runtime_error("qwen3_store_kvcache_ expects k_cache and v_cache to have the same shape.");
    }
    if (indices->ndim() != 1 || indices->shape()[0] != k->shape()[0]) {
        throw std::runtime_error("qwen3_store_kvcache_ expects indices shape [num_tokens].");
    }
    const auto src_row = row_size(k);
    const auto cache_row = row_size(k_cache);
    if (src_row != cache_row) {
        throw std::runtime_error("qwen3_store_kvcache_ source/cache row size mismatch.");
    }
    if (!k->is_contiguous() || !v->is_contiguous() || !k_cache->is_contiguous() || !v_cache->is_contiguous()) {
        throw std::runtime_error("qwen3_store_kvcache_ expects contiguous k, v, k_cache, and v_cache tensors.");
    }

    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
    auto k_at = infinicore::adaptor::to_aten_tensor(k).reshape({static_cast<int64_t>(k->shape()[0]), static_cast<int64_t>(src_row)});
    auto v_at = infinicore::adaptor::to_aten_tensor(v).reshape({static_cast<int64_t>(v->shape()[0]), static_cast<int64_t>(src_row)});
    auto k_cache_at = infinicore::adaptor::to_aten_tensor(k_cache).reshape({static_cast<int64_t>(k_cache->shape()[0]), static_cast<int64_t>(cache_row)});
    auto v_cache_at = infinicore::adaptor::to_aten_tensor(v_cache).reshape({static_cast<int64_t>(v_cache->shape()[0]), static_cast<int64_t>(cache_row)});
    auto indices_at = infinicore::adaptor::to_aten_tensor(indices);

    k_cache_at.index_copy_(0, indices_at, k_at);
    v_cache_at.index_copy_(0, indices_at, v_at);
#else
    (void)k;
    (void)v;
    (void)k_cache;
    (void)v_cache;
    (void)indices;
    throw std::runtime_error("qwen3_store_kvcache_ requires an ATen-enabled NVIDIA build.");
#endif
}

} // namespace infinicore::op

