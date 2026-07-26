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

inline py::tuple py_deepseek_v4_flashmla_sparse_attention_with_metadata_(Tensor q,
                                                                         Tensor raw_cache,
                                                                         Tensor indices,
                                                                         Tensor topk_lengths,
                                                                         py::object attn_sink,
                                                                         Tensor output,
                                                                         py::object tile_scheduler_metadata,
                                                                         py::object num_splits,
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
    std::optional<Tensor> tile_scheduler_metadata_tensor = std::nullopt;
    if (!tile_scheduler_metadata.is_none()) {
        tile_scheduler_metadata_tensor = tile_scheduler_metadata.cast<Tensor>();
    }
    std::optional<Tensor> num_splits_tensor = std::nullopt;
    if (!num_splits.is_none()) {
        num_splits_tensor = num_splits.cast<Tensor>();
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
    auto schedule = op::deepseek_v4_flashmla_sparse_attention_with_metadata_(q,
                                                                            raw_cache,
                                                                            indices,
                                                                            topk_lengths,
                                                                            attn_sink_tensor,
                                                                            output,
                                                                            tile_scheduler_metadata_tensor,
                                                                            num_splits_tensor,
                                                                            softmax_scale,
                                                                            page_size,
                                                                            head_dim_v,
                                                                            extra_raw_cache_tensor,
                                                                            extra_indices_tensor,
                                                                            extra_topk_lengths_tensor,
                                                                            extra_page_size);
    return py::make_tuple(output, schedule.tile_scheduler_metadata, schedule.num_splits);
}

inline void bind_deepseek_v4_flashmla_compute(py::module &m) {


    m.def("deepseek_v4_compress_fused_norm_rope_",
          &op::deepseek_v4_compress_fused_norm_rope_,
          py::arg("input"),
          py::arg("norm_weight"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"),
          R"doc(Default fused norm plus last-64 RoPE for DeepSeek-V4 compressed attention.)doc");
    m.def("deepseek_v4_compress_fused_norm_rope_naive_",
          &op::deepseek_v4_compress_fused_norm_rope_naive_,
          py::arg("input"),
          py::arg("norm_weight"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"));
    m.def("deepseek_v4_compress_fused_norm_rope_kernel_",
          &op::deepseek_v4_compress_fused_norm_rope_kernel_,
          py::arg("input"),
          py::arg("norm_weight"),
          py::arg("epsilon"),
          py::arg("freqs_cis"),
          py::arg("positions"));



    m.def("deepseek_v4_c4_compress_stateful",
          &op::deepseek_v4_c4_compress_stateful,
          py::arg("kv_score_input"),
          py::arg("ape"),
          py::arg("compressor_state"),
          py::arg("write_loc"),
          py::arg("extra_loc"),
          py::arg("positions"),
          R"doc(Default stateful C4 compression for DeepSeek-V4 compressed attention.)doc");
    m.def("deepseek_v4_c4_compress_stateful_naive",
          &op::deepseek_v4_c4_compress_stateful_naive,
          py::arg("kv_score_input"),
          py::arg("ape"),
          py::arg("compressor_state"),
          py::arg("write_loc"),
          py::arg("extra_loc"),
          py::arg("positions"));
    m.def("deepseek_v4_c4_compress_stateful_kernel",
          &op::deepseek_v4_c4_compress_stateful_kernel,
          py::arg("kv_score_input"),
          py::arg("ape"),
          py::arg("compressor_state"),
          py::arg("write_loc"),
          py::arg("extra_loc"),
          py::arg("positions"));



    m.def("deepseek_v4_c128_compress_stateful",
          &op::deepseek_v4_c128_compress_stateful,
          py::arg("kv_score_input"),
          py::arg("ape"),
          py::arg("compressor_state"),
          py::arg("write_loc"),
          py::arg("positions"),
          R"doc(Default stateful C128 compression for DeepSeek-V4 compressed attention.)doc");
    m.def("deepseek_v4_c128_compress_stateful_naive",
          &op::deepseek_v4_c128_compress_stateful_naive,
          py::arg("kv_score_input"),
          py::arg("ape"),
          py::arg("compressor_state"),
          py::arg("write_loc"),
          py::arg("positions"));
    m.def("deepseek_v4_c128_compress_stateful_kernel",
          &op::deepseek_v4_c128_compress_stateful_kernel,
          py::arg("kv_score_input"),
          py::arg("ape"),
          py::arg("compressor_state"),
          py::arg("write_loc"),
          py::arg("positions"));

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

    m.def("deepseek_v4_flashmla_sparse_attention_with_metadata_",
          &ops::py_deepseek_v4_flashmla_sparse_attention_with_metadata_,
          py::arg("q"),
          py::arg("raw_cache"),
          py::arg("indices"),
          py::arg("topk_lengths"),
          py::arg("attn_sink"),
          py::arg("output"),
          py::arg("tile_scheduler_metadata"),
          py::arg("num_splits"),
          py::arg("softmax_scale"),
          py::arg("page_size") = 256,
          py::arg("head_dim_v") = 512,
          py::arg("extra_raw_cache") = py::none(),
          py::arg("extra_indices") = py::none(),
          py::arg("extra_topk_lengths") = py::none(),
          py::arg("extra_page_size") = 0,
          R"doc(DeepSeek-V4 sparse/radix FlashMLA attention with reusable scheduler metadata.)doc");
}

} // namespace infinicore::ops
