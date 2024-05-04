cmake_minimum_required(VERSION 3.4...3.27)
project(cpp_model LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
# Maybe you have to do something linke this: sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
message(STATUS "Eigen3 version ${EIGEN3_VERSION} found in ${EIGEN3_INCLUDE_DIR}")

find_package(pybind11 REQUIRED)
pybind11_add_module(cpp_model wrapper.cpp cpp_model.hpp) # Add wrapper code and all library source files

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensions off

set_target_properties(cpp_model
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}"
)

# Setting this does not seem to speed up the programm, maybe for larger calculations
# target_compile_options(cpp_model PRIVATE -Werror -Wall -Wextra -pedantic -std=c++20 -DNDEBUG -O3 -ffast-math -fno-math-errno -march=native -mfma -msse2)

message(STATUS "Cpp project dir: ${PROJECT_SOURCE_DIR}")
#FILE(WRITE ${CMAKE_BINARY_DIR}/__init__.py) # Allows to import the created python library using import in a python script
