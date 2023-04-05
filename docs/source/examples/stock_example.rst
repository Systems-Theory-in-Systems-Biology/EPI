Stock Data Model
----------------
A high-dimensional stock data model is contained in :file:`epipy/examples/stock/stock.py`.

Specialities
____________

* External Data Source: The model loads stock data from the web.
* High-Dimensional: The model has a high number of dimensions: data_dim = 19, param_dim = 6. The samples emcee strongly recommended to use at least 12 walkers for this model. 
* Automatic Differentiation: The derivatives are calculated automatically with jax by deriving from the class :py:class:`~epipy.core.model.JaxModel`,
  which automatically calculates sets :py:meth:`~epipy.core.model.Model.jacobian`.
* JIT compilation: Inheriting from :py:class:`~epipy.core.model.JaxModel` also enables jit compilation / optimization for the forward and jacobian method.
  This usually results in a significant execution speedup. It also allows to run your model on the gpu.

.. literalinclude:: ../../../epipy/examples/stock/stock.py
  :language: python
  :pyobject: Stock
