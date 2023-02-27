import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit, vmap

from epi.core.model import ArtificialModelInterface, JaxModel, Model


class JaxPlant(JaxModel, ArtificialModelInterface):
    """A plant model which inherits from the JaxModel to define the jacobian
    Param0: Water [0,1]
    Param1: Sun   [0,1]
    Data0: Size [0,2] # the more water and sun the better
    Data1: Health [0,1], to much water is not good, too much sun is not good
    Data2: Sciarid :P
    """

    paramDim = 2
    dataDim = 3

    @classmethod
    def forward(cls, param):
        return jnp.array(
            [
                param[0] * param[1],
                jnp.prod(jnp.sin(jnp.pi * param)),
                jnp.exp(param[0]) - 0.999,
            ]
        )

    def getCentralParam(self) -> np.ndarray:
        return np.array([0.5, 0.5])

    def getParamSamplingLimits(self) -> np.ndarray:
        return np.array([[0.0, 1.0], [0.0, 1.0]])

    def generateArtificialData(self, numSamples=1000):
        # randomly create true parameters in [0,1]x[0,1]
        trueParamSample = np.random.rand(numSamples, 2)

        artificialData = vmap(self.forward, in_axes=0)(trueParamSample)

        np.savetxt(
            f"Data/{self.getModelName()}Data.csv",
            artificialData,
            delimiter=",",
        )
        np.savetxt(
            f"Data/{self.getModelName()}Params.csv",
            trueParamSample,
            delimiter=",",
        )


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

    paramDim = 2
    dataDim = 3

    def forward(self, param):
        return fw(param)

    def jacobian(self, param):
        return bw(param)

    def getCentralParam(self) -> np.ndarray:
        return np.array([0.5, 0.5])

    def getParamSamplingLimits(self) -> np.ndarray:
        return np.array([[0.0, 1.0], [0.0, 1.0]])

    def generateArtificialData(
        self, numSamples=ArtificialModelInterface.NUM_ARTIFICIAL_SAMPLES
    ):
        # randomly create true parameters in [0,1]x[0,1]
        trueParamSample = np.random.rand(numSamples, 2)

        artificialData = vmap(self.forward, in_axes=0)(trueParamSample)

        np.savetxt(
            f"Data/{self.getModelName()}Data.csv",
            artificialData,
            delimiter=",",
        )
        np.savetxt(
            f"Data/{self.getModelName()}Params.csv",
            trueParamSample,
            delimiter=",",
        )
