Usage Examples 
==============

In this section several examples how the EPI can be used will be presented.

* Running the 1D-Temperature model from example models
* High-Dimensional Stock Data
* Using your own simple model
* Creating a model using SBML
* Custom C++ Class



Temperature Model
-----------------
The temperature model is contained in :code:`epic/example_models/temperature`.
Now describe the simplest way to run epi on it and then the simples way to visualize results.
Describe where results are stored.


StockData Model
---------------
The stock data model is contained in :code:`epic/example_models/stock`.
Overwrite dataLoader and implements both interfaces, describe them.
Describe how to vis the high dim data

Own simple model
----------------
Use a really simple model with explicit jacobian


SBML Model
----------
Show how to create a model from an sbml file with sbmllib


C++ Model
---------
Show how to create a C++ model for really expensive model evaluations
We will need new category cplusplus deps: cmake, pybind11
