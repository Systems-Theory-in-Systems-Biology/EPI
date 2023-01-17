StockData Model
---------------
The high-dimensional stock data model is contained in :code:`epic/example_models/applications/stock`.

.. TODO::

    The model implements the function :math:`y_i(q_i)=????`. And why high number of walkers? see TODO

Specialities
____________

* External Data Source: The model shows how to use an external data source in the workflow by overwriting the method :py:meth:`epic.core.model.Model.dataLoader`.
* High-Dimensional: The model has a high number of dimensions: DataDim = 19, ParamDim = 6
  * Number of Walkers in MCMC Sampling: Requires a large number of walkers, because TODO!!!
  * Visualization: The visualizataion can be done for each dimension seperately, for two selected dimensions or using spider web plots.

.. TODO::

    Show the visualization somewhere or implement it as standard / enum. Show code call here?

.. literalinclude:: ../../epic/example_models/applications/stock.py
  :language: python
  :pyobject: Stock
