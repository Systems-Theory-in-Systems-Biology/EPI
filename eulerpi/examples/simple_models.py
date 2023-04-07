from typing import Optional

import jax.numpy as jnp
import numpy as np

from eulerpi.core.model import ArtificialModelInterface, JaxModel


class Linear(JaxModel, ArtificialModelInterface):
    param_dim = 2
    data_dim = 2

    PARAM_LIMITS = np.array([[-0.2, 1.2], [-0.2, 1.2]])
    CENTRAL_PARAM = np.array([0.5, 0.5])

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(central_param, param_limits, name=name, **kwargs)

    @classmethod
    def forward(cls, param):
        return jnp.array([param[0] * 10, (-2.0) * param[1] - 2.0])

    def generate_artificial_params(self, num_samples: int):
        return np.random.rand(num_samples, self.param_dim)


class Exponential(JaxModel):
    param_dim = 2
    data_dim = 2

    PARAM_LIMITS = np.array([[1.0, 2.0], [1.0, 2.0]])
    CENTRAL_PARAM = np.array([1.0, 1.0])

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(central_param, param_limits, name=name, **kwargs)

    @classmethod
    def forward(cls, param):
        return jnp.array([param[0] * jnp.exp(1), jnp.exp(param[1])])


class LinearODE(JaxModel, ArtificialModelInterface):
    param_dim = 2
    data_dim = 2

    PARAM_LIMITS = np.array([[-2.0, 4.0], [-2.0, 4.0]])
    CENTRAL_PARAM = np.array([1.5, 1.5])

    TRUE_PARAM_LIMITS = np.array([[1.0, 2.0], [1.0, 2.0]])

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(central_param, param_limits, name=name, **kwargs)

    @classmethod
    def forward(cls, param):
        return jnp.array(
            [
                param[0] * jnp.exp(param[1] * 1.0),
                param[0] * jnp.exp(param[1] * 2.0),
            ]
        )

    def generate_artificial_params(self, num_samples: int):
        return np.random.rand(num_samples, 2) + 1
