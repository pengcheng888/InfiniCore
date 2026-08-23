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


    m.def("deepseek_v4_store_flashmla_raw_cache_",
          &op::deepseek_v4_store_flashmla_raw_cache_,
          py::arg("input"),
          py::arg("cache"),
          py::arg("indices"),
          py::arg("page_size") = 256,
          R"doc(DeepSeek-V4 SGLang-compatible raw FlashMLA SWA cache store.)doc");
    m.def("deepseek_v4_store_flashmla_raw_cache_naive_",
          &op::deepseek_v4_store_flashmla_raw_cache_naive_,
          py::arg("input"),
          py::arg("cache"),
          py::arg("indices"),
          py::arg("page_size") = 256);
    m.def("deepseek_v4_store_flashmla_raw_cache_kernel_",
          &op::deepseek_v4_store_flashmla_raw_cache_kernel_,
          py::arg("input"),
          py::arg("cache"),
          py::arg("indices"),
          py::arg("page_size") = 256);



    m.def("deepseek_v4_indexer_rotate_",
          &op::deepseek_v4_indexer_rotate_,
          py::arg("input"),
          py::arg("apply_scale") = true,
          R"doc(DeepSeek-V4 SGLang-compatible C4 indexer Hadamard rotate.)doc");
    m.def("deepseek_v4_indexer_rotate_naive_",
          &op::deepseek_v4_indexer_rotate_naive_,
          py::arg("input"),
          py::arg("apply_scale") = true);
    m.def("deepseek_v4_indexer_rotate_128_kernel_",
          &op::deepseek_v4_indexer_rotate_128_kernel_,
          py::arg("input"),
          py::arg("apply_scale") = true);



    m.def("deepseek_v4_store_indexer_raw_cache_",
          &op::deepseek_v4_store_indexer_raw_cache_,
          py::arg("input"),
          py::arg("cache"),
          py::arg("indices"),
          py::arg("page_size") = 64,
          R"doc(DeepSeek-V4 SGLang-compatible raw C4 indexer cache store.)doc");
    m.def("deepseek_v4_store_indexer_raw_cache_naive_",
          &op::deepseek_v4_store_indexer_raw_cache_naive_,
          py::arg("input"),
          py::arg("cache"),
          py::arg("indices"),
          py::arg("page_size") = 64);
    m.def("deepseek_v4_store_indexer_raw_cache_kernel_",
          &op::deepseek_v4_store_indexer_raw_cache_kernel_,
          py::arg("input"),
          py::arg("cache"),
          py::arg("indices"),
          py::arg("page_size") = 64);



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
