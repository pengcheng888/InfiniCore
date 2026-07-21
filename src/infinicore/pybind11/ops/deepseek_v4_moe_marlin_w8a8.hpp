#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_moe_marlin_w8a8.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_moe_marlin_w8a8(py::module &m) {
    m.def("deepseek_v4_moe_marlin_w8a8_",
          &op::deepseek_v4_moe_marlin_w8a8_,
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
          py::arg("mode"),
          py::arg("delta"),
          R"doc(DeepSeek-V4 AITER MoE Marlin W8A8 bridge.)doc");
    m.def("deepseek_v4_moe_marlin_w8a8_fp8_",
          &op::deepseek_v4_moe_marlin_w8a8_fp8_,
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
          py::arg("mode"),
          py::arg("delta"),
          R"doc(DeepSeek-V4 AITER MoE Marlin W8A8 FP8 bridge.)doc");
}

} // namespace infinicore::ops
