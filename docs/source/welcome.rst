.. image:: images/epi.png
   :width: 200pt

-------------------------------
EPI - Euler Parameter Inference
-------------------------------


The Euler Parameter Inference Codebase (EPI) is a python package for inverse parameter inference.
The EPI algorithm takes observed data and a model as input and returns a parameter distribution, which is consistent with the observed data by solving the inverse problem directly. In the case of a one-to-one mapping, this is the true underlying distribution.
We support SBML ode models and user provided models.

.. Put the badges here?

------------
Installation
------------

The package is no yet available on pypi.
..  and can be installed with: 

.. .. code-block:: bash
   
..    pip install epi

You can build the library from the newest source code by following the :doc:`Development Quickstart Guide </MarkdownLinks/development>`.

------------
How to start
------------

| Derive your model from :py:class:`epi.core.model.Model` and implement the abstract functions :py:meth:`~epi.core.model.Model.forward` and :py:meth:`~epi.core.model.Model.jacobian`.
| You also need to define the data and parameter Dimension, :py:attr:`~epi.core.model.Model.data_dim` and :py:attr:`~epi.core.model.Model.param_dim` as class attributes or property methods.

.. code-block:: python

    import jax.numpy as jnp

    from epi.core.model import Model

    class MyModel(Model):

        param_dim = N # The dimension of a parameter point
        data_dim = M # The dimension of a data point

        def forward(self, param):
            return jnp.array(...)

        def jacobian(self, param):
            return jnp.array(...)

To evaluate the model and infer the parameter distribution, call:

.. code-block:: python

    from epi.core.inference import inference

    from my_model import MyModel

    central_param = np.array([0.5, -1.5, ...])
    param_limits = np.array([[0.0, 1.0], [-3.0, 0.0], ...])

    model = MyModel(central_param, param_limits)
    inference(model=model, data="my_data.csv")

The parameter :py:attr:`data` can be a numpy-2d-array or a PathLike object, which leads to a csv file. In the shown case, the csv file :file:`my_data.csv` has to contain the data in the format

.. code-block:: text

    datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
    datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
    datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
    ...
    datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN

which corresponds to a matrix with the shape :py:attr:`nSamples` x :py:attr:`data_dim`. More available options and parameters for the :py:mod:`~epi.core.inference` method can be found in the documentation.
Most importantly the inference can be done with grid based methods (dense grids, sparse grids) or sampling methods (mcmc).

The results are stored in the locations

* :file:`./Applications/<ModelName>/.../OverallParams.csv`
* :file:`./Applications/<ModelName>/.../OverallSimResults.csv`
* :file:`./Applications/<ModelName>/.../OverallDensityEvals.csv`

and contain the sampled parameters or grid points, the corresponding data points obtained from the model forward pass and the corresponding density evaluation.

.. note::
   
   Please read the documentation for our :doc:`Examples </examples>`.
