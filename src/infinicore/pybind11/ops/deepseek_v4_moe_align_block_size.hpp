#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_moe_align_block_size.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_moe_align_block_size(py::module &m) {
    m.def("deepseek_v4_moe_align_block_size_",
          &op::deepseek_v4_moe_align_block_size_,
          py::arg("topk_ids"),
          py::arg("num_experts"),
          py::arg("block_size"),
          py::arg("sorted_token_ids"),
          py::arg("experts_ids"),
          py::arg("num_tokens_post_pad"),
          py::arg("cumsum_buffer"),
          py::arg("pad_sorted_token_ids") = false,
          R"doc(DeepSeek-V4 MoE align-block-size backed by SGLang sgl_kernel.)doc");
}

} // namespace infinicore::ops
