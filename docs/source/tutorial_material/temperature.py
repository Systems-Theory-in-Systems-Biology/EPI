from typing import Optional

import jax.numpy as jnp
import numpy as np

from eulerpi.models import BaseModel


class Temperature(BaseModel):

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
