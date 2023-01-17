Example Models
==============

TODO: Move most of this into the welcome page and the sphinx documentation?

EPI can be used in several ways and can be applied to many different problems.
We will show how EPI can be applied to a simple, one-dimensional problem; a high-dimensional problem;
a sbml model; and a problem defined through external C++ code.
For more details on the mathematics behind the epi algorithm, we refer to our published paper.
For more details on how to use EPI, we refer to our Tutorial page.

Content
-------

* :ref:`1D-Temperature model from the tutorial<Temperature Model>`
* :ref:`High-Dimensional Stock Data<StockData Model>`
* :ref:`Creating a model using SBML<SBML Model>`
* :ref:`External C++ Model<C++ Model>`

General
-----------------
In all exaples we assume a given (discrete) data probability distribution :math:`y_i \sim Y`.
The goal is to find the parameter probability distribution :math:`Q` satisfying :math:`Y = s(Q)`.

Temperature Model
-----------------
The temperature model is contained in :code:`epic/example_models/temperature`.
The model :math:`y_i(q_i)=60 \cos(q_i)-30=s(q_i)` describes the temperature for a place on the earth :math:`y_i` by using the latitude coordinates :math:`q_i`.


There are two possible scenarios:

* Artificial Data:
* Real Data

Now describe the simplest way to run epi on it and then the simples way to visualize results.
Describe where results are stored.


StockData Model
---------------
The stock data model is contained in :code:`epic/example_models/stock`.
Overwrite dataLoader and implements both interfaces, describe them.
Describe how to vis the high dim data

SBML Model
----------
Show how to create a model from an sbml file with sbmllib
You can visualize sbml files with https://sbml4humans.de/.


C++ Model
---------
Many libraries in the field of scientific computing are written in C++
to achieve fast code execution and do not have python bindings. The C++ model example shows how you can
call C++ code from your python :py:class:`~epic.core.model.Model` class.
It is primarily intened for fast implementations of the `forward` and `jacobian` method.

Preparation
___________

.. code-block::
    pip install pybind11
    sudo apt install cmake
    sudo apt install pybind11
    sudo apt install eigen3-dev
    sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen

C++ Model Definition
____________________

Include cpp model file directly.

The example code is inconsistent in the following way:
It uses a normal array for the forward method,
but an eigen vector as input and an eigen matrix as output
for the jacobian method. This allows to show us how to write wrapper code
for normal arrays as well as for eigen objects. On the python side exlusively
numpy 1d/2d arrays will be used.

.. TODO::
    What about vectorization? Jax will likely not help with speeding up.

.. note::
    PyBind11 Documentation: https://pybind11.readthedocs.io/en/stable/
    PyBind11 Eigen: https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html


Compilation
___________

.. code-block::
    cd /epic/example_models/cpp/
    mkdir build && cd build
    cmake ..
    make -j

You can use the example model as template for your own C++ Model.

Python Side Model
_________________

include python side model
