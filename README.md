<!-- # Euler Parameter Inference -->
<h1></h1>

![EPI](epi.png "logo")

<!-- The badges we want to display -->
[![pages-build-deployment](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/pages/pages-build-deployment)
[![Build & Publish Documentation](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/sphinx.yml/badge.svg)](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/sphinx.yml)
[![CI](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/ci.yml/badge.svg)](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/ci.yml)

The Euler Parameter Inference (EPI) is a python package for inverse parameter inference.

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

    paramDim = N # The dimension of a parameter point
    dataDim = M # The dimension of a data point

    def forward(self, param):
        return jnp.array(...)

    def getParamSamplingLimits(self):
        return jnp.array([[-1.,1.], [-101.1, 13.4],...]) # [[UpperBound_dim1,LowerBound_dim1],...]

    def getCentralParam(self):
        return jnp.array([0.5, -30.0,...])

    def jacobian(self, param):
        return jnp.array(...)
```

To evaluate the model and infer the parameter distribution, call:

```python
from epi.sampling import inference

from my_model import MyModel

model = MyModel()
inference(model=model, dataPath="my_data.csv")
```

The file `my_data.csv` has to contain the data in csv format with `seperator=,` in the format

```text
datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
...
datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
```

which corresponds to a matrix with the shape `nSamples x dataDim`.

## Documentation

The full documentation to this software, including a detailed tutorial on how to use EPI and the api documentation, can be found under [Documentation](https://Systems-Theory-in-Systems-Biology.github.io/EPI/).
