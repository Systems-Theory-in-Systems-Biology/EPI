import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit

from epi.core.model import ArtificialModelInterface, JaxModel, Model


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

    defaultParamSamplingLimits = np.array([[0, 1], [0, 1]])
    defaultcentral_param = np.array([0.5, 0.5])

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


class ExternalPlant(Model, ArtificialModelInterface):
    """A plant model which uses functions defined outside the class to evaluate the forward pass and the jacobian
    Param0: Water [0,1]
    Param1: Sun   [0,1]
    Data0: Size [0,2] # the more water and sun the better
    Data1: Health [0,1], to much water is not good, too much sun is not good
    Data2: Trauerfliegen :P
    """

    param_dim = 2
    data_dim = 3

    defaultParamSamplingLimits = np.array([[0, 1], [0, 1]])
    defaultcentral_param = np.array([0.5, 0.5])

    def forward(self, param):
        return fw(param)

    def jacobian(self, param):
        return bw(param)

    def generate_artificial_params(self, num_samples: int):
        return np.random.rand(num_samples, 2)
