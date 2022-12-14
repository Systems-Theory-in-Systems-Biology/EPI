#ifndef MY_CPP_MODEL
#define MY_CPP_MODEL

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


// Returned from py::array_t<double>.request()
// struct buffer_info {
//     void *ptr;
//     size_t itemsize;
//     std::string format;
//     int ndim;
//     std::vector<size_t> shape;
//     std::vector<size_t> strides;
// };


namespace py = pybind11;

double forward(double param) {
    return param;
}

double jacobian(double param) {
    return 1.;
}

py::array_t<double> forward(py::array_t<double> param) {
    py::buffer_info bufParam = param.request();

    // Storage for the result
    auto res = py::array_t<double>(bufParam.size);
    py::buffer_info bufRes = res.request();


    double* ptrRes = (double*)bufRes.ptr;
    double* ptrParam = (double*)bufParam.ptr;

    for (int i = 0; i < bufParam.shape[0]; i++)
    {
        ptrRes[i] = ptrParam[i] * i;
    }

    return result
}

py::array_t<double> param jacobian(py::array_t<double> param) {
    py::buffer_info bufParam = param.request();

    // Storage for the result
    auto res = py::array_t<double>(bufParam**2.size);
    py::buffer_info bufRes = res.request();


    //Apply resources
    auto res = py::array_t<double>(buf1.size);
    //Resize to 2d array
    result.resize({buf1.shape[0],buf1.shape[1]});
        double* ptrRes = (double*)bufRes.ptr;


    return 1.;
}

#endif
