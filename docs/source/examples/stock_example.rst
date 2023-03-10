stock_data Model
---------------
A high-dimensional stock data model is contained in :file:`epi/examples/stock/stock.py`.

Specialities
____________

* External Data Source: The model loads stock data from the web.
* High-Dimensional: The model has a high number of dimensions: data_dim = 19, param_dim = 6. The samples emcee strongly recommended to use at least 12 walkers for this model. 

.. literalinclude:: ../../../epi/examples/stock/stock.py
  :language: python
  :pyobject: Stock
