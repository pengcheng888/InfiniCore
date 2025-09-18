#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <infinicore.hpp>

namespace py = pybind11;

namespace infinicore {

PYBIND11_MODULE(infinicore, m) {
    py::enum_<DataType>(m, "dtype")
        .value("bfloat16", DataType::BF16)
        .value("float16", DataType::F16)
        .value("float32", DataType::F32)
        .value("float64", DataType::F64)
        .value("int32", DataType::I32)
        .value("int64", DataType::I64)
        .value("uint8", DataType::U8)
        .export_values();

    py::class_<Device>(m, "Device")
        .def(py::init<const Device::Type &, const Device::Index &>(),
             py::arg("type"), py::arg("index") = 0)
        .def_property_readonly("type", &Device::getType)
        .def_property_readonly("index", &Device::getIndex)
        .def("__repr__", static_cast<std::string (Device::*)() const>(&Device::toString));

    py::class_<Tensor>(m, "Tensor")
        .def_static("empty", &Tensor::empty,
                    py::arg("shape"), py::arg("dtype"), py::arg("device"))
        .def_static("zeros", &Tensor::zeros,
                    py::arg("shape"), py::arg("dtype"), py::arg("device"))
        .def_static("ones", &Tensor::ones,
                    py::arg("shape"), py::arg("dtype"), py::arg("device"))

        .def("shape", [](const Tensor &self) {
            return self->shape();
        })
        .def("dtype", [](const Tensor &self) {
            return self->dtype();
        })
        .def("device", [](const Tensor &self) {
            return self->device();
        });
}

} // namespace infinicore
