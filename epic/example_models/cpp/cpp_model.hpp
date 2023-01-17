#ifndef MY_CPP_MODEL
#define MY_CPP_MODEL

#include <cmath>

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

// Mapping from 
py::array_t<double> forward(py::array_t<double> param) {
    py::buffer_info bufParam = param.request();

    // Storage for the result
    py::array_t<double> result = py::array(py::buffer_info(
        nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
        sizeof(double),     /* Size of one item */
        py::format_descriptor<double>::value, /* Buffer format */
        1,          /* How many dimensions? */
        { 3 },  /* Number of elements for each dimension */
        { sizeof(double) }  /* Strides for each dimension */
    ));
    py::buffer_info bufRes = result.request();

    double * ptrParam = static_cast<double *>(bufParam.ptr);
    double * ptrRes   = static_cast<double *>(bufRes.ptr);

    double water = ptrParam[0];
    double sun   = ptrParam[1];
    
    double size  = water * sun;
    double green = std::sin(M_PI * water) * std::sin(M_PI * sun);
    double flies = std::exp(water)-0.999;
    
    ptrRes[0] = size;
    ptrRes[1] = green;
    ptrRes[2] = flies;

    return result;
}

//TODO: Use this type to avoid copy? Measure runtime for larger problem.
//using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

typedef Eigen::Matrix<double, 3, 2> Matrix32d;
//Jacobian for f:R^n -> R^m is Jf:R^m -> R^n = R^mxn
Matrix32d jacobian(Eigen::Vector2d param) {
    auto res = Matrix32d(3,2);
    res(0,0) = param(1);
    res(0,1) = param(0);
    res(1,0) = M_PI * std::cos(M_PI * param(0)) * std::sin(M_PI * param(1));
    res(1,1) = std::sin(M_PI * param(0)) * M_PI * std::cos(M_PI * param(1));
    res(2,0) = param(0) * std::exp(param(0));
    res(2,1) = 0.0;
    return res;
}

#endif
