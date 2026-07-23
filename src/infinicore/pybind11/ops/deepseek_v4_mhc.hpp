#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_mhc.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_mhc(py::module &m) {
    m.def("deepseek_v4_mhc_pre_naive_",
          &op::deepseek_v4_mhc_pre_naive_,
          py::arg("y"),
          py::arg("post"),
          py::arg("comb"),
          py::arg("x"),
          py::arg("fn"),
          py::arg("scale"),
          py::arg("base"),
          py::arg("rms_eps"),
          py::arg("hc_eps"),
          py::arg("sinkhorn_iters"));
    m.def("deepseek_v4_mhc_pre_kernel_",
          &op::deepseek_v4_mhc_pre_kernel_,
          py::arg("y"),
          py::arg("post"),
          py::arg("comb"),
          py::arg("x"),
          py::arg("fn"),
          py::arg("scale"),
          py::arg("base"),
          py::arg("rms_eps"),
          py::arg("hc_eps"),
          py::arg("sinkhorn_iters"));
    m.def("deepseek_v4_mhc_post_naive_",
          &op::deepseek_v4_mhc_post_naive_,
          py::arg("y"),
          py::arg("x"),
          py::arg("residual"),
          py::arg("post"),
          py::arg("comb"));
    m.def("deepseek_v4_mhc_post_kernel_",
          &op::deepseek_v4_mhc_post_kernel_,
          py::arg("y"),
          py::arg("x"),
          py::arg("residual"),
          py::arg("post"),
          py::arg("comb"));
    m.def("deepseek_v4_mhc_head_naive_",
          &op::deepseek_v4_mhc_head_naive_,
          py::arg("y"),
          py::arg("x"),
          py::arg("fn"),
          py::arg("scale"),
          py::arg("base"),
          py::arg("rms_eps"),
          py::arg("hc_eps"));
    m.def("deepseek_v4_mhc_head_kernel_",
          &op::deepseek_v4_mhc_head_kernel_,
          py::arg("y"),
          py::arg("x"),
          py::arg("fn"),
          py::arg("scale"),
          py::arg("base"),
          py::arg("rms_eps"),
          py::arg("hc_eps"));
    m.def("deepseek_v4_moe_w8a8_naive_",
          &op::deepseek_v4_moe_w8a8_naive_,
          py::arg("y"),
          py::arg("x"),
          py::arg("topk_weights"),
          py::arg("topk_indices"),
          py::arg("w13"),
          py::arg("w13_scale"),
          py::arg("w2"),
          py::arg("w2_scale"),
          py::arg("swiglu_limit"));
}

} // namespace infinicore::ops
