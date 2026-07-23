#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_flashmla_compute.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void py_deepseek_v4_flashmla_sparse_attention_(Tensor q,
                                                      Tensor raw_cache,
                                                      Tensor indices,
                                                      Tensor topk_lengths,
                                                      py::object attn_sink,
                                                      Tensor output,
                                                      float softmax_scale,
                                                      int page_size,
                                                      int head_dim_v,
                                                      py::object extra_raw_cache,
                                                      py::object extra_indices,
                                                      py::object extra_topk_lengths,
                                                      int extra_page_size) {
    std::optional<Tensor> attn_sink_tensor = std::nullopt;
    if (!attn_sink.is_none()) {
        attn_sink_tensor = attn_sink.cast<Tensor>();
    }
    std::optional<Tensor> extra_raw_cache_tensor = std::nullopt;
    if (!extra_raw_cache.is_none()) {
        extra_raw_cache_tensor = extra_raw_cache.cast<Tensor>();
    }
    std::optional<Tensor> extra_indices_tensor = std::nullopt;
    if (!extra_indices.is_none()) {
        extra_indices_tensor = extra_indices.cast<Tensor>();
    }
    std::optional<Tensor> extra_topk_lengths_tensor = std::nullopt;
    if (!extra_topk_lengths.is_none()) {
        extra_topk_lengths_tensor = extra_topk_lengths.cast<Tensor>();
    }
    op::deepseek_v4_flashmla_sparse_attention_(q,
                                               raw_cache,
                                               indices,
                                               topk_lengths,
                                               attn_sink_tensor,
                                               output,
                                               softmax_scale,
                                               page_size,
                                               head_dim_v,
                                               extra_raw_cache_tensor,
                                               extra_indices_tensor,
                                               extra_topk_lengths_tensor,
                                               extra_page_size);
}

inline void bind_deepseek_v4_flashmla_compute(py::module &m) {
    m.def("deepseek_v4_flashmla_sparse_attention_",
          &ops::py_deepseek_v4_flashmla_sparse_attention_,
          py::arg("q"),
          py::arg("raw_cache"),
          py::arg("indices"),
          py::arg("topk_lengths"),
          py::arg("attn_sink"),
          py::arg("output"),
          py::arg("softmax_scale"),
          py::arg("page_size") = 256,
          py::arg("head_dim_v") = 512,
          py::arg("extra_raw_cache") = py::none(),
          py::arg("extra_indices") = py::none(),
          py::arg("extra_topk_lengths") = py::none(),
          py::arg("extra_page_size") = 0,
          R"doc(DeepSeek-V4 sparse/radix FlashMLA-compatible attention over packed raw SWA/cache plus optional C4/C128 extra cache.)doc");
}

} // namespace infinicore::ops
