#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <infinicore.hpp>

namespace py = pybind11;

namespace infinicore {

PYBIND11_MODULE(infinicore, m) {
    py::enum_<DataType>(m, "dtype")
        .value("bfloat16", DataType::bfloat16)
        .value("float16", DataType::float16)
        .value("float32", DataType::float32)
        .value("float64", DataType::float64)
        .value("int32", DataType::int32)
        .value("int64", DataType::int64)
        .value("uint8", DataType::uint8)
        .export_values();

    py::class_<Device>(m, "Device")
        .def(py::init<const Device::Type &, const Device::Index &>(),
             py::arg("type"), py::arg("index") = 0)
        .def_property_readonly("type", &Device::get_type)
        .def_property_readonly("index", &Device::get_index)
        .def("__repr__", static_cast<std::string (Device::*)() const>(&Device::to_string));

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
