.. image:: images/epi.png
   :width: 100pt

-------------------------------
EPI - Euler Parameter Inference
-------------------------------

Euler Parameter Inference (EPI) is a Python package for inverse parameter inference. It provides an implementation of the EPI algorithm, which takes observed data and a model as input and returns a parameter distribution consistent with the observed data by solving the inverse problem directly. In the case of a one-to-one mapping, this is the true underlying distribution.

.. Put the badges here?
.. image:: https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/pages/pages-build-deployment/badge.svg
    :target: https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/pages/pages-build-deployment
.. image:: https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/sphinx.yml/badge.svg
    :target: https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/sphinx.yml
.. image:: https://img.shields.io/github/actions/workflow/status/Systems-Theory-in-Systems-Biology/EPI/ci.yml?label=pytest&logo=pytest
    :target: https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/ci.yml
    :alt: pytest

.. image:: https://img.shields.io/badge/flake8-checked-blue.svg
    :target: https://flake8.pycqa.org/
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: ./LICENSE.md
.. image:: https://img.shields.io/badge/python-3.10-purple.svg
    :target: https://www.python.org/
.. image:: https://img.shields.io/pypi/v/eulerpi
    :target: https://pypi.org/project/eulerpi/

--------
Features
--------

EPI supports

* SBML ode models
* User provided models
* Models with automatic differentation using jax

------------
Installation
------------

The package is available on pypi and can be installed with: 

.. code-block:: bash
   
   pip install eulerpi

or

.. code-block:: bash

    pip install eulerpi[sbml]

for the support of sbml models.

Make sure that you have the following C++ libraries installed:

.. code-block:: bash
   
   sudo apt install -y swig libblas-dev libatlas-base-dev libhdf5-dev

You can also build the library from the latest source code by following the :doc:`Development Quickstart Guide </MarkdownLinks/development>`.

------------
How to start
------------

.. To use EPI, derive your model from :py:class:`eulerpi.model.BaseModel` and implement the abstract functions :py:meth:`~eulerpi.model.BaseModel.forward` and :py:meth:`~eulerpi.model.BaseModel.jacobian`. You also need to define the data and parameter dimension, :py:attr:`~eulerpi.model.BaseModel.data_dim` and :py:attr:`~eulerpi.model.BaseModel.param_dim` of your model.

To use EPI, derive your model from the BaseModel class and implement the abstract functions. Here's an example code snippet:

.. code-block:: python
    
    # my_model.py
    import jax.numpy as jnp

    from eulerpi.model import BaseModel

    class MyModel(BaseModel):

        param_dim = N # The dimension of a parameter point
        data_dim = M # The dimension of a data point

        def forward(self, param):
            return jnp.array(...)

        def jacobian(self, param):
            return jnp.array(...)

To evaluate the model and infer the parameter distribution, call:

.. code-block:: python

    from eulerpi import inference

    from my_model import MyModel

    # This line is needed for multiprocessing in python
    if __name__ == "__main__":
        central_param = np.array([0.5, -1.5, ...])
        param_limits = np.array([[0.0, 1.0], [-3.0, 0.0], ...])

        model = MyModel(central_param, param_limits)
        inference(model=model, data="my_data.csv")

The parameter :py:attr:`data` can be a numpy-2d-array or a PathLike object that points to a CSV file. In the example shown above, the CSV file :file:`my_data.csv` should contain the data in the following format:

.. code-block:: text

    datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
    datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
    datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
    ...
    datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN

This corresponds to a matrix with the shape :py:attr:`nSamples` x :py:attr:`data_dim`. For more available options and parameters for the :py:mod:`~eulerpi.inference` method, please refer to the API documentation.
Note that the inference can be done with grid-based methods (dense grids, sparse grids) or sampling methods (mcmc).

The results are stored in the following locations

* :file:`./Applications/<ModelName>/.../OverallParams.csv`
* :file:`./Applications/<ModelName>/.../OverallSimResults.csv`
* :file:`./Applications/<ModelName>/.../OverallDensityEvals.csv`

These files contain the sampled parameters, the corresponding data points obtained from the model forward pass, and the corresponding density evaluation.

.. note::
   
   Please read the documentation for our :doc:`Examples </examples>`.
