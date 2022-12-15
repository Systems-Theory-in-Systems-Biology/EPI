// pybind11_wrapper.cpp
#include <pybind11/pybind11.h>
#include "cpp_model.hpp"

PYBIND11_MODULE(cpp_model, m) {
    m.doc() = "C++ EPI Model implemented with pybind11"; // Optional module docstring
    m.def("forward", &forward, "A function that calculates the forward pass");
    m.def("jacobian", &jacobian, "A function that calculates the jacobian");
}
