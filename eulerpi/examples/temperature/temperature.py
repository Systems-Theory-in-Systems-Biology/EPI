import importlib
from typing import Optional

import jax.numpy as jnp
import numpy as np

from eulerpi.core.model import ArtificialModelInterface, Model

# from functools import partial
# from jax import jit


class Temperature(Model):
    """ """

    param_dim = 1
    data_dim = 1

    PARAM_LIMITS = np.array([[0, np.pi / 2]])
    CENTRAL_PARAM = np.array([np.pi / 4.0])

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(central_param, param_limits, name=name, **kwargs)

    def forward(self, param):
        low_T = -30.0
        high_T = 30.0
        res = jnp.array(
            [low_T + (high_T - low_T) * jnp.cos(jnp.abs(param[0]))]
        )
        return res

    def jacobian(self, param):
        return jnp.array([60.0 * jnp.sin(jnp.abs(param[0]))])


class TemperatureArtificial(Temperature, ArtificialModelInterface):
    def generate_artificial_params(self, num_data_points: int = -1):
        paramPath = importlib.resources.path(
            "eulerpi.examples.temperature", "TemperatureArtificialParams.csv"
        )
        true_param_sample = np.loadtxt(paramPath, delimiter=",", ndmin=2)
        return true_param_sample


class TemperatureWithFixedParams(Temperature):
    def __init__(
        self,
        low_T: np.double = -30.0,
        high_T: np.double = 30.0,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.low_T = low_T
        self.high_T = high_T

    def forward(self, param):
        return self.calc_forward(param, self.high_T, self.low_T)

    def calc_forward(self, param, high_T, low_T):
        res = jnp.array(
            [low_T + (high_T - low_T) * jnp.cos(jnp.abs(param[0]))]
        )
        return res

    def jacobian(self, param):
        return self.calc_jacobian(param, self.high_T, self.low_T)

    def calc_jacobian(self, param, high_T, low_T):
        return jnp.array([(high_T - low_T) * jnp.sin(jnp.abs(param[0]))])
