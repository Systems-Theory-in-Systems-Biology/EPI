C++ Model
---------
Many libraries in the field of scientific computing are written in C++
to achieve fast code execution and do not have python bindings. The C++ model example shows how you can
call C++ code from your python :py:class:`~epic.core.model.Model` class.
It is primarily intended for fast implementations of the `forward` and `jacobian` method.

Specialities
____________

* Calling C++ Code: Calls external c++ code using pybind11
* Performance Comparison: The file :file:`python_reference_plants.py` includes
  python models implementing the same mapping. You can compare the performance of the different approaches.

Preparation
___________

.. code-block:: bash

    sudo apt install cmake
    sudo apt install pybind11
    sudo apt install eigen3-dev
    sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen

C++ Model Definition
____________________

.. literalinclude:: ../../epic/example_models/cpp/cpp_plant.py
  :language: c++

.. TODO::

    Why is pygments not parsing the c++ code?

The example code is inconsistent in the following way:
It uses a normal array for the forward method,
but an eigen vector as input and an eigen matrix as output
for the jacobian method. This allows to show us how to write wrapper code
for normal arrays as well as for eigen objects. On the python side exclusively
numpy 1d/2d arrays will be used.

.. TODO::

    What about vectorization? Jax will likely not help with speeding up.

.. note::

    For more information on how to use pybind11 see:
    PyBind11 Documentation: https://pybind11.readthedocs.io/en/stable/
    PyBind11 Eigen Notes: https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html


Compilation
___________

.. code-block:: bash

    cd /epic/example_models/cpp/
    mkdir build && cd build
    cmake ..
    make -j

Python Side Model
_________________

.. literalinclude:: ../../epic/example_models/cpp/cpp_plant.py
    :language: python

You can use the example model as template for your own C++ Model.
