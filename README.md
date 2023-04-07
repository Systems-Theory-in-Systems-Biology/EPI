<!-- # Euler Parameter Inference -->
<h1></h1>

![EPI](https://github.com/Systems-Theory-in-Systems-Biology/EPI/blob/main/epi.png?raw=True "logo")

<!-- The badges we want to display -->
[![pages-build-deployment](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/pages/pages-build-deployment)
[![Build & Publish Documentation](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/sphinx.yml/badge.svg)](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/sphinx.yml)
[![CI](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/ci.yml/badge.svg)](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/ci.yml)
[![pytest](https://img.shields.io/github/actions/workflow/status/Systems-Theory-in-Systems-Biology/EPI/ci.yml?label=pytest&logo=pytest)](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/ci.yml)

[![flake8](https://img.shields.io/badge/flake8-checked-blue.svg)](https://flake8.pycqa.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE.md)
[![Python](https://img.shields.io/badge/python-3.10-purple.svg)](https://www.python.org/)

Euler Parameter Inference (EPI) is a Python package for inverse parameter inference. It provides an implementation of the EPI algorithm, which takes observed data and a model as input and returns a parameter distribution consistent with the observed data by solving the inverse problem directly. In the case of a one-to-one mapping, this is the true underlying distribution.

## Documentation

The full documentation to this software, including a detailed tutorial on how to use EPI and the api documentation, can be found under [Documentation](https://Systems-Theory-in-Systems-Biology.github.io/EPI/).

## Features

EPI supports

- SBML ode models
- User provided models
- Models with automatic differentation using jax

## Installation

The package is available on pypi and can be installed with:

```bash
pip install eulerpi
```

Make sure that you have the following C++ libraries installed

```bash
sudo apt install -y swig libblas-dev libatlas-base-dev
```

You can also build the library from the latest source code by following the [Development Quickstart Guide](./DEVELOPMENT.md#quickstart).

## Using the library

To use EPI, derive your model from the `Model` class and implement the abstract functions. Here's an example code snippet:

```python
# my_model.py

import jax.numpy as jnp

from eulerpi.core.model import Model

class MyModel(Model):

    param_dim = N # The dimension of a parameter point
    data_dim = M # The dimension of a data point

    def forward(self, param):
        return jnp.array(...)

    def jacobian(self, param):
        return jnp.array(...)
```

To evaluate the model and infer the parameter distribution, call:

```python
from eulerpi.sampling import inference

from my_model import MyModel

central_param = np.array([0.5, -1.5, ...])
param_limits = np.array([[0.0, 1.0], [-3.0, 0.0], ...])

model = MyModel(central_param, param_limits)
inference(model=model, data="my_data.csv")
```

The `data` argument can be a numpy-2d-array or a PathLike object that points to a CSV file. In the example shown above, the CSV file `my_data.csv` should contain the data in the following format:

```text
datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
...
datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
```

This corresponds to a matrix with the shape `nSamples x data_dim`. For more available options and parameters for the `inference` method, please refer to the [api documentation](https://systems-theory-in-systems-biology.github.io/EPI/eulerpi.core.html#module-eulerpi.core.inference). Note that the inference can be done with grid-based methods (dense grids, sparse grids) or sampling methods (mcmc).

The results are stored in the following location:

* `./Applications/<ModelName>/.../OverallParams.csv`
* `./Applications/<ModelName>/.../OverallSimResults.csv`
* `./Applications/<ModelName>/.../OverallDensityEvals.csv`

These files contain the sampled parameters, the corresponding data points obtained from the model forward pass, and the corresponding density evaluation.
