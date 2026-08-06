#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_compressor_kv_score.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_compressor_kv_score(py::module &m) {
    m.def("deepseek_v4_compressor_kv_score_packed_",
          &op::deepseek_v4_compressor_kv_score_packed_,
          py::arg("out"),
          py::arg("input"),
          py::arg("wkv_gate"));
    m.def("deepseek_v4_compressor_kv_score_unpacked_",
          &op::deepseek_v4_compressor_kv_score_unpacked_,
          py::arg("out"),
          py::arg("input"),
          py::arg("wkv"),
          py::arg("wgate"));
}

} // namespace infinicore::ops
