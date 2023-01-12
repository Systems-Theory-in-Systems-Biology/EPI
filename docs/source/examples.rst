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


C++ Model
---------
Show how to create a C++ model for really expensive model evaluations
We will need new category cplusplus deps: cmake, pybind11
