#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_deep_gemm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_deep_gemm(py::module &m) {
    m.def("deepseek_v4_deep_gemm_low_latency_grouped_gemm_",
          &op::deepseek_v4_deep_gemm_low_latency_grouped_gemm_,
          py::arg("matrix_a"),
          py::arg("matrix_b"),
          py::arg("matrix_a_scale"),
          py::arg("matrix_b_scale"),
          py::arg("actual_tokens"),
          py::arg("matrix_c"),
          py::arg("max_tokens"),
          py::arg("experts"),
          py::arg("cu_s"),
          py::arg("block_wise"),
          py::arg("b_overlap") = false,
          py::arg("signal") = py::none(),
          R"doc(DeepSeek-V4 DeepGEMM low-latency grouped GEMM bridge.)doc");
    m.def("deepseek_v4_deep_gemm_moe_w8a8_i8_marlin_prefill_down_",
          &op::deepseek_v4_deep_gemm_moe_w8a8_i8_marlin_prefill_down_,
          py::arg("input"),
          py::arg("b_qweight"),
          py::arg("output"),
          py::arg("a_scale"),
          py::arg("b_scale"),
          py::arg("topk_weights"),
          py::arg("sorted_token_ids"),
          py::arg("expert_ids"),
          py::arg("num_tokens_post_pad"),
          py::arg("top_k"),
          py::arg("real_topk"),
          R"doc(DeepSeek-V4 DeepGEMM MoE W8A8 I8 Marlin prefill-down bridge.)doc");
    m.def("deepseek_v4_deep_gemm_moe_w8a8_marlin_decode_down_fp8_",
          &op::deepseek_v4_deep_gemm_moe_w8a8_marlin_decode_down_fp8_,
          py::arg("input"),
          py::arg("b_qweight"),
          py::arg("output"),
          py::arg("a_scale"),
          py::arg("b_scale"),
          py::arg("topk_weights"),
          py::arg("sorted_token_ids"),
          py::arg("expert_ids"),
          py::arg("num_tokens_post_pad"),
          py::arg("top_k"),
          py::arg("real_topk"),
          R"doc(DeepSeek-V4 DeepGEMM MoE W8A8 Marlin decode-down FP8 bridge.)doc");
}

} // namespace infinicore::ops
