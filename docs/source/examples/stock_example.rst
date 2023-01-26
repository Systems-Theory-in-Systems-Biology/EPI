StockData Model
---------------
The high-dimensional stock data model is contained in :file:`epi/examples/stock/stock.py`.

.. TODO::

    The model implements the function :math:`y_i(q_i)`

Specialities
____________

* External Data Source: The model shows how to use an external data source in the workflow by overwriting the method :py:meth:`epi.core.model.Model.dataLoader`.
* High-Dimensional: The model has a high number of dimensions: DataDim = 19, ParamDim = 6
  * Large Number of Walkers in MCMC Sampling
  * Visualization: The visualization can be done for each dimension separately, for two selected dimensions or using spider web plots.

.. TODO::

    Visualization

.. literalinclude:: ../../../epi/examples/stock/stock.py
  :language: python
  :pyobject: Stock
