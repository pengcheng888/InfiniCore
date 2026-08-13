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

inline void py_deepseek_v4_flashmla_sparse_attention_out_workspace_(Tensor q,
                                                                    Tensor raw_cache,
                                                                    Tensor indices,
                                                                    Tensor topk_lengths,
                                                                    py::object attn_sink,
                                                                    Tensor output,
                                                                    Tensor lse,
                                                                    Tensor lse_accum,
                                                                    Tensor o_accum,
                                                                    Tensor tile_scheduler_metadata,
                                                                    Tensor num_splits,
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
    op::deepseek_v4_flashmla_sparse_attention_out_workspace_(q,
                                                             raw_cache,
                                                             indices,
                                                             topk_lengths,
                                                             attn_sink_tensor,
                                                             output,
                                                             lse,
                                                             lse_accum,
                                                             o_accum,
                                                             tile_scheduler_metadata,
                                                             num_splits,
                                                             softmax_scale,
                                                             page_size,
                                                             head_dim_v,
                                                             extra_raw_cache_tensor,
                                                             extra_indices_tensor,
                                                             extra_topk_lengths_tensor,
                                                             extra_page_size);
}

inline void py_deepseek_v4_flashmla_sparse_attention_metadata_(Tensor tile_scheduler_metadata,
                                                              Tensor num_splits,
                                                              Tensor topk_lengths,
                                                              int topk,
                                                              py::object extra_topk_lengths,
                                                              int extra_topk) {
    std::optional<Tensor> extra_topk_lengths_tensor = std::nullopt;
    if (!extra_topk_lengths.is_none()) {
        extra_topk_lengths_tensor = extra_topk_lengths.cast<Tensor>();
    }
    op::deepseek_v4_flashmla_sparse_attention_metadata_(tile_scheduler_metadata,
                                                       num_splits,
                                                       topk_lengths,
                                                       topk,
                                                       extra_topk_lengths_tensor,
                                                       extra_topk);
}

inline py::object py_deepseek_v4_flashmla_cuda_call(const char *name, py::args args, py::kwargs kwargs) {
    py::gil_scoped_acquire gil;
    py::object fn = py::module_::import("flash_mla.cuda").attr(name);
    if (kwargs && PyDict_Size(kwargs.ptr()) > 0) {
        return fn(*args, **kwargs);
    }
    return fn(*args);
}

inline void bind_deepseek_v4_flashmla_cuda_entry(py::module &m, const char *deepseek_name, const char *flashmla_name) {
    m.def(deepseek_name,
          [flashmla_name](py::args args, py::kwargs kwargs) {
              return py_deepseek_v4_flashmla_cuda_call(flashmla_name, args, kwargs);
          },
          py::return_value_policy::move);
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

    m.def("deepseek_v4_flashmla_sparse_attention_out_workspace_",
          &ops::py_deepseek_v4_flashmla_sparse_attention_out_workspace_,
          py::arg("q"),
          py::arg("raw_cache"),
          py::arg("indices"),
          py::arg("topk_lengths"),
          py::arg("attn_sink"),
          py::arg("output"),
          py::arg("lse"),
          py::arg("lse_accum"),
          py::arg("o_accum"),
          py::arg("tile_scheduler_metadata"),
          py::arg("num_splits"),
          py::arg("softmax_scale"),
          py::arg("page_size") = 256,
          py::arg("head_dim_v") = 512,
          py::arg("extra_raw_cache") = py::none(),
          py::arg("extra_indices") = py::none(),
          py::arg("extra_topk_lengths") = py::none(),
          py::arg("extra_page_size") = 0,
          R"doc(DeepSeek-V4 FlashMLA sparse attention using caller-owned output, lse, split-KV workspaces, and scheduler metadata.)doc");

    m.def("deepseek_v4_flashmla_sparse_attention_metadata_",
          &ops::py_deepseek_v4_flashmla_sparse_attention_metadata_,
          py::arg("tile_scheduler_metadata"),
          py::arg("num_splits"),
          py::arg("topk_lengths"),
          py::arg("topk"),
          py::arg("extra_topk_lengths") = py::none(),
          py::arg("extra_topk") = -1,
          R"doc(Refresh caller-owned DeepSeek-V4 sparse FlashMLA decode scheduler metadata.)doc");

    bind_deepseek_v4_flashmla_cuda_entry(m, "deepseek_v4_dense_decode_fwd", "dense_decode_fwd");
    bind_deepseek_v4_flashmla_cuda_entry(m, "deepseek_v4_dense_decode_fwd_kvfp8", "dense_decode_fwd_kvfp8");
    bind_deepseek_v4_flashmla_cuda_entry(m, "deepseek_v4_dense_decode_fwd_qkvfp8", "dense_decode_fwd_qkvfp8");
    bind_deepseek_v4_flashmla_cuda_entry(m, "deepseek_v4_fwd_kvcache_mla_fp8", "fwd_kvcache_mla_fp8");
    bind_deepseek_v4_flashmla_cuda_entry(m, "deepseek_v4_fwd_kvcache_mla_fp8_with_cat", "fwd_kvcache_mla_fp8_with_cat");
    bind_deepseek_v4_flashmla_cuda_entry(m, "deepseek_v4_fwd_kvcache_mla_nope_pe", "fwd_kvcache_mla_nope_pe");
    bind_deepseek_v4_flashmla_cuda_entry(m, "deepseek_v4_fwd_kvcache_quantization_mla", "fwd_kvcache_quantization_mla");
    bind_deepseek_v4_flashmla_cuda_entry(m, "deepseek_v4_fwd_kvcache_quantization_q_nope_pe_mla", "fwd_kvcache_quantization_q_nope_pe_mla");
    bind_deepseek_v4_flashmla_cuda_entry(m, "deepseek_v4_get_mla_decoding_metadata_dense_fp8", "get_mla_decoding_metadata_dense_fp8");
    bind_deepseek_v4_flashmla_cuda_entry(m, "deepseek_v4_sparse_decode_fwd", "sparse_decode_fwd");
    bind_deepseek_v4_flashmla_cuda_entry(m, "deepseek_v4_sparse_prefill_fwd", "sparse_prefill_fwd");
}

} // namespace infinicore::ops
