#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "device.hpp"
#include "dtype.hpp"

namespace infinicore {

PYBIND11_MODULE(infinicore, m) {
    pybind11::class_<py::DataType>(m, "dtype")
        .def("__repr__", py::DataType::toString);

    m.attr("float32") = py::DataType{DataType::F32};
    m.attr("float") = m.attr("float32");
    m.attr("float64") = py::DataType{DataType::F64};
    m.attr("double") = m.attr("float64");
    m.attr("complex32") = py::DataType{DataType::C32};
    m.attr("chalf") = m.attr("complex32");
    m.attr("complex64") = py::DataType{DataType::C64};
    m.attr("cfloat") = m.attr("complex64");
    m.attr("complex128") = py::DataType{DataType::C128};
    m.attr("cdouble") = m.attr("complex128");
    m.attr("float16") = py::DataType{DataType::F16};
    m.attr("half") = m.attr("float16");
    m.attr("bfloat16") = py::DataType{DataType::BF16};
    m.attr("uint8") = py::DataType{DataType::U8};
    m.attr("int8") = py::DataType{DataType::I8};
    m.attr("int16") = py::DataType{DataType::I16};
    m.attr("short") = m.attr("int16");
    m.attr("int32") = py::DataType{DataType::I32};
    m.attr("int") = m.attr("int32");
    m.attr("int64") = py::DataType{DataType::I64};
    m.attr("long") = m.attr("int64");
    m.attr("bool") = py::DataType{DataType::BOOL};

    pybind11::class_<py::Device>(m, "device")
        .def(pybind11::init<const std::string &, Device::Index>(),
             pybind11::arg("type"), pybind11::arg("index"))
        .def_property_readonly("type", &py::Device::getType)
        .def_property_readonly("index", &py::Device::getIndex)
        .def("__repr__", &py::Device::toRepresentation)
        .def("__str__", &py::Device::toString);

    pybind11::class_<Tensor>(m, "Tensor")
        .def_static("empty", &Tensor::empty,
                    pybind11::arg("shape"), pybind11::arg("dtype"), pybind11::arg("device"), pybind11::arg("pin_memory") = false)
        .def_static("zeros", &Tensor::zeros,
                    pybind11::arg("shape"), pybind11::arg("dtype"), pybind11::arg("device"), pybind11::arg("pin_memory") = false)
        .def_static("ones", &Tensor::ones,
                    pybind11::arg("shape"), pybind11::arg("dtype"), pybind11::arg("device"), pybind11::arg("pin_memory") = false)

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
