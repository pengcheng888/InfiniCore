#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_flashmla_cache.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void py_deepseek_v4_flashmla_cache_indexer_(Tensor req_to_token,
                                                   Tensor req_pool_indices,
                                                   Tensor page_kernel_lens,
                                                   py::object kv_start_idx,
                                                   Tensor kv_indices,
                                                   int req_to_token_stride,
                                                   int kv_indices_stride,
                                                   int page_size) {
    std::optional<Tensor> kv_start_idx_tensor = std::nullopt;
    if (!kv_start_idx.is_none()) {
        kv_start_idx_tensor = kv_start_idx.cast<Tensor>();
    }
    op::deepseek_v4_flashmla_cache_indexer_(req_to_token,
                                            req_pool_indices,
                                            page_kernel_lens,
                                            kv_start_idx_tensor,
                                            kv_indices,
                                            req_to_token_stride,
                                            kv_indices_stride,
                                            page_size);
}

inline void bind_deepseek_v4_flashmla_cache(py::module &m) {
    m.def("deepseek_v4_fused_store_flashmla_cache_",
          &op::deepseek_v4_fused_store_flashmla_cache_,
          py::arg("kv_c"),
          py::arg("k_pe"),
          py::arg("kv_cache"),
          py::arg("slot_mapping"),
          py::arg("kv_cache_dtype"),
          py::arg("scale"));

    m.def("deepseek_v4_flashmla_cache_indexer_",
          &ops::py_deepseek_v4_flashmla_cache_indexer_,
          py::arg("req_to_token"),
          py::arg("req_pool_indices"),
          py::arg("page_kernel_lens"),
          py::arg("kv_start_idx"),
          py::arg("kv_indices"),
          py::arg("req_to_token_stride"),
          py::arg("kv_indices_stride"),
          py::arg("page_size") = 64);
}

} // namespace infinicore::ops
