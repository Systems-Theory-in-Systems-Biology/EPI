import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit

from eulerpi.models import ArtificialModelInterface, BaseModel, JaxModel


class JaxPlant(JaxModel, ArtificialModelInterface):
    """A plant model which inherits from the JaxModel to define the jacobian
    Param0: Water [0,1]
    Param1: Sun   [0,1]
    Data0: Size [0,2] # the more water and sun the better
    Data1: Health [0,1], to much water is not good, too much sun is not good
    Data2: Sciarid :P

    """

    param_dim = 2
    data_dim = 3

    PARAM_LIMITS = np.array([[0, 1], [0, 1]])
    CENTRAL_PARAM = np.array([0.5, 0.5])

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: str = None,
        **kwargs,
    ) -> None:
        super().__init__(
            central_param,
            param_limits,
            name,
            **kwargs,
        )

    @classmethod
    def forward(cls, param):
        return jnp.array(
            [
                param[0] * param[1],
                jnp.prod(jnp.sin(jnp.pi * param)),
                jnp.exp(param[0]) - 0.999,
            ]
        )

    def generate_artificial_params(self, num_samples: int):
        return np.random.rand(num_samples, 2)


@jit
def fw(param):
    return jnp.array(
        [
            param[0] * param[1],
            jnp.prod(jnp.sin(jnp.pi * param)),
            jnp.exp(param[0]) - 0.999,
        ]
    )


fwJac = jit(jacrev(fw))


@jit
def bw(param):
    return fwJac(param)


class ExternalPlant(BaseModel, ArtificialModelInterface):
    """A plant model which uses functions defined outside the class to evaluate the forward pass and the jacobian
    Param0: Water [0,1]
    Param1: Sun   [0,1]
    Data0: Size [0,2] # the more water and sun the better
    Data1: Health [0,1], to much water is not good, too much sun is not good
    Data2: Trauerfliegen :P
    """

    param_dim = 2
    data_dim = 3

    PARAM_LIMITS = np.array([[0, 1], [0, 1]])
    CENTRAL_PARAM = np.array([0.5, 0.5])

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: str = None,
    ) -> None:
        super().__init__(
            central_param,
            param_limits,
            name,
        )

    def forward(self, param):
        return fw(param)

    def jacobian(self, param):
        return bw(param)

    def generate_artificial_params(self, num_samples: int):
        return np.random.rand(num_samples, 2)
