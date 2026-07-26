#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_mhc.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_mhc(py::module &m) {
    m.def("deepseek_v4_mhc_pre_",
          &op::deepseek_v4_mhc_pre_,
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
    m.def("deepseek_v4_mhc_post_",
          &op::deepseek_v4_mhc_post_,
          py::arg("y"),
          py::arg("x"),
          py::arg("residual"),
          py::arg("post"),
          py::arg("comb"));
    m.def("deepseek_v4_mhc_head_",
          &op::deepseek_v4_mhc_head_,
          py::arg("y"),
          py::arg("x"),
          py::arg("fn"),
          py::arg("scale"),
          py::arg("base"),
          py::arg("rms_eps"),
          py::arg("hc_eps"));

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

    m.def("deepseek_v4_mhc_pre_naive",
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
    m.def("deepseek_v4_mhc_pre_kernel",
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
    m.def("deepseek_v4_mhc_post_naive",
          &op::deepseek_v4_mhc_post_naive_,
          py::arg("y"),
          py::arg("x"),
          py::arg("residual"),
          py::arg("post"),
          py::arg("comb"));
    m.def("deepseek_v4_mhc_post_kernel",
          &op::deepseek_v4_mhc_post_kernel_,
          py::arg("y"),
          py::arg("x"),
          py::arg("residual"),
          py::arg("post"),
          py::arg("comb"));
    m.def("deepseek_v4_mhc_head_naive",
          &op::deepseek_v4_mhc_head_naive_,
          py::arg("y"),
          py::arg("x"),
          py::arg("fn"),
          py::arg("scale"),
          py::arg("base"),
          py::arg("rms_eps"),
          py::arg("hc_eps"));
    m.def("deepseek_v4_mhc_head_kernel",
          &op::deepseek_v4_mhc_head_kernel_,
          py::arg("y"),
          py::arg("x"),
          py::arg("fn"),
          py::arg("scale"),
          py::arg("base"),
          py::arg("rms_eps"),
          py::arg("hc_eps"));
}

} // namespace infinicore::ops
