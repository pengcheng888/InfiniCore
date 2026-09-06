#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/flash_mla/get_mla_metadata.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_flash_mla_get_mla_metadata(py::module &m) {
    m.def("get_mla_metadata",
          &op::flash_mla::get_mla_metadata,
          R"doc(Return an empty FlashMLA scheduler metadata object.)doc");
}

} // namespace infinicore::ops
