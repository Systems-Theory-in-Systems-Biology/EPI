<!-- # Euler Parameter Inference Codebase -->
<h1></h1>

![EPIC](epic.png "logo")

<!-- The badges we want to display -->
[![pages-build-deployment](https://github.com/Systems-Theory-in-Systems-Biology/EPIC/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/Systems-Theory-in-Systems-Biology/EPIC/actions/workflows/pages/pages-build-deployment)
[![CI](https://github.com/Systems-Theory-in-Systems-Biology/EPIC/actions/workflows/ci.yml/badge.svg)](https://github.com/Systems-Theory-in-Systems-Biology/EPIC/actions/workflows/ci.yml)
The Euler Parameter Inference Codebase (EPIC) is a python package for inverse parameter inference.

## About

The EPI algorithm takes observed data and a model as input and returns a parameter distribution, which is consistent with the observed data by solving the inverse problem directly. In the case of a one-to-one mapping, this is the true underlying distribution.

We support SBML ode models and user provided models.

## Installation

The package is available on pypi and can be installed with:\
```pip install epic```\
You can also build the library from the newest source code by following the [Development Quickstart Guide](./DEVELOPMENT.md#quickstart).

## Using the library

Derive your model from ```Model``` class and implement the abstract functions. Optionally you can also implement the abstract functions from ```ArtificialModelInterface``` and ```VisualizationModelInterface```.

```python
import jax.numpy as jnp

from epic.core.model import Model

class MyModel(Model):
    def forward(self, param):
        return jnp.array(...)

    def getParamSamplingLimits(self):
        return np.array([[-1.,1.], [-101.1, 13.4],...]) # [[UpperBound_dim1,LowerBound_dim1],...]

    def getCentralParam(self):
        return np.array([0.5, -30.0,...])

    def jacobian(self, param):
        return jnp.array(...)
```

To evaluate the model and infer the parameter distribution, call:

```python
model.inference("my_data.csv")
```

## Documentation

The full documentation to this software, including a detailed tutorial on how to use EPI, can be found under [Documentation](https://Systems-Theory-in-Systems-Biology.github.io/EPIC/).
