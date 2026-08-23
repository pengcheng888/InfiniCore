#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_compress_stateful.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_compress_stateful(py::module &m) {
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
}

} // namespace infinicore::ops
