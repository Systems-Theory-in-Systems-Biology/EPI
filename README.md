<!-- # Euler Parameter Inference -->
<h1></h1>

![EPI](epi.png "logo")

<!-- The badges we want to display -->
[![pages-build-deployment](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/pages/pages-build-deployment)
[![Build & Publish Documentation](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/sphinx.yml/badge.svg)](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/sphinx.yml)
[![CI](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/ci.yml/badge.svg)](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/ci.yml)
[![pytest](https://img.shields.io/github/actions/workflow/status/Systems-Theory-in-Systems-Biology/EPI/ci.yml?label=pytest&logo=pytest)](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/ci.yml)

[![flake8](https://img.shields.io/badge/flake8-checked-blue.svg)](https://flake8.pycqa.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE.md)
[![Python](https://img.shields.io/badge/python-3.10-purple.svg)](https://www.python.org/)

The Euler Parameter Inference (EPI) is a python package for inverse parameter inference.

## Documentation

The full documentation to this software, including a detailed tutorial on how to use EPI and the api documentation, can be found under [Documentation](https://Systems-Theory-in-Systems-Biology.github.io/EPI/).

## About

The EPI algorithm takes observed data and a model as input and returns a parameter distribution, which is consistent with the observed data by solving the inverse problem directly. In the case of a one-to-one mapping, this is the true underlying distribution.

We support SBML ode models and user provided models.

## Installation

  ---
  **IMPORTANT**

  The package is not yet available on pypi.

  <!-- ```text
  pip install epi
  ``` -->

  ---

You can build the library from the newest source code by following the [Development Quickstart Guide](./DEVELOPMENT.md#quickstart).

## Using the library

Derive your model from ```Model``` class and implement the abstract functions.

```python
import jax.numpy as jnp

from epi.core.model import Model

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
from epi.sampling import inference

from my_model import MyModel

central_param = np.array([0.5, -1.5, ...])
param_limits = np.array([[0.0, 1.0], [-3.0, 0.0], ...])

model = MyModel(central_param, param_limits)
inference(model=model, data="my_data.csv")
```

`data` can be a numpy-2d-array or a PathLike object, which leads to a csv file. In the shown case, the csv file `my_data.csv` has to contain the data in the format

```text
datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
...
datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
```

which corresponds to a matrix with the shape `nSamples x data_dim`. More available options and parameters for the `inference` method can be found in the [api documentation](https://systems-theory-in-systems-biology.github.io/EPI/epi.core.html#module-epi.core.inference). Most importantly the inference can be done with grid based methods (dense grids, sparse grids) or sampling methods (mcmc).

The results are stored in the following location:

* `./Applications/<ModelName>/.../OverallParams.csv`
* `./Applications/<ModelName>/.../OverallSimResults.csv`
* `./Applications/<ModelName>/.../OverallDensityEvals.csv`

and contain the sampled parameters, the corresponding data points obtained from the model forward pass and the corresponding density evaluation.
