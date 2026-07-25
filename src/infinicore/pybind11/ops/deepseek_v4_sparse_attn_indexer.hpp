#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_sparse_attn_indexer.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void py_deepseek_v4_sparse_attn_indexer_prefill_(Tensor q,
                                                        Tensor k,
                                                        Tensor weights,
                                                        Tensor cu_seqlen_ks,
                                                        Tensor cu_seqlen_ke,
                                                        Tensor logits,
                                                        Tensor topk_indices,
                                                        py::object kv_scale,
                                                        int topk_tokens,
                                                        bool clean_logits) {
    std::optional<Tensor> kv_scale_tensor = std::nullopt;
    if (!kv_scale.is_none()) {
        kv_scale_tensor = kv_scale.cast<Tensor>();
    }
    op::deepseek_v4_sparse_attn_indexer_prefill_(q,
                                                 k,
                                                 weights,
                                                 cu_seqlen_ks,
                                                 cu_seqlen_ke,
                                                 logits,
                                                 topk_indices,
                                                 kv_scale_tensor,
                                                 topk_tokens,
                                                 clean_logits);
}

inline void bind_deepseek_v4_sparse_attn_indexer(py::module &m) {
    m.def("deepseek_v4_sparse_attn_indexer_prefill_",
          &ops::py_deepseek_v4_sparse_attn_indexer_prefill_,
          py::arg("q"),
          py::arg("k"),
          py::arg("weights"),
          py::arg("cu_seqlen_ks"),
          py::arg("cu_seqlen_ke"),
          py::arg("logits"),
          py::arg("topk_indices"),
          py::arg("kv_scale") = py::none(),
          py::arg("topk_tokens") = 2048,
          py::arg("clean_logits") = true);

    m.def("deepseek_v4_sparse_attn_indexer_decode_",
          &op::deepseek_v4_sparse_attn_indexer_decode_,
          py::arg("q"),
          py::arg("fused_kv_cache"),
          py::arg("weights"),
          py::arg("context_lens"),
          py::arg("block_table"),
          py::arg("schedule_meta"),
          py::arg("logits"),
          py::arg("topk_indices"),
          py::arg("max_context_len"),
          py::arg("next_n"),
          py::arg("topk_tokens") = 2048,
          py::arg("clean_logits") = true);

    m.def("deepseek_v4_c4_sparse_attn_indexer_",
          &op::deepseek_v4_c4_sparse_attn_indexer_,
          py::arg("q"),
          py::arg("indexer_weights"),
          py::arg("indexer_kv_cache_raw"),
          py::arg("c4_seq_lens"),
          py::arg("page_table"),
          py::arg("logits"),
          py::arg("out_page_indices"),
          py::arg("max_c4_seq_len"),
          py::arg("page_size") = 64,
          py::arg("weight_scale") = 1.0f,
          py::arg("clean_logits") = false);

    m.def("deepseek_v4_c4_act_quant_fused_scale_kernel_",
          &op::deepseek_v4_c4_act_quant_fused_scale_kernel_,
          py::arg("q"),
          py::arg("indexer_weights"),
          py::arg("q_fp8"),
          py::arg("q_scale"),
          py::arg("fused_weights"),
          py::arg("weight_scale") = 1.0f);

    m.def("deepseek_v4_topk_transform_512_kernel_",
          &op::deepseek_v4_topk_transform_512_kernel_,
          py::arg("scores"),
          py::arg("seq_lens"),
          py::arg("page_table"),
          py::arg("out_page_indices"),
          py::arg("page_size") = 64);
}

} // namespace infinicore::ops
