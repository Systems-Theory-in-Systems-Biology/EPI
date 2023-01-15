import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit

from epic.core.model import ArtificialModelInterface, Model


@jit
def fw(param):
    return jnp.array(
        [
            param[0] * param[1],
            jnp.prod(jnp.sin(jnp.pi * param)),
            jnp.exp(param[0]) - 0.999,
        ]
    )


fw_jac = jit(jacrev(fw))


@jit
def bw(param):
    return fw_jac(param)


class Plant(Model, ArtificialModelInterface):
    """A plant model which uses a c++ library with eigen3 to evaluate the forward pass and the gradient
    Param0: Water [0,1]
    Param1: Sun   [0,1]
    Data0: Size [0,2] # the more water and sun the better
    Data1: Health [0,1], to much water is not good, too much sun is not good
    Data2: Trauerfliegen :P
    """

    def forward(self, param):
        return fw(param)

    def jacobian(self, param):
        return bw(param)

    def getCentralParam(self) -> np.ndarray:
        return np.array([0.5, 0.5])

    def getParamSamplingLimits(self) -> np.ndarray:
        return np.array([[0.0, 1.0], [0.0, 1.0]])

    def generateArtificialData(self, numSamples=1000):
        # randomly create true parameters in [0,1]x[0,1]
        trueParamSample = np.random.rand(numSamples, 2)

        artificialData = np.zeros((trueParamSample.shape[0], 3))
        for i in range(trueParamSample.shape[0]):
            artificialData[i, :] = self.forward(trueParamSample[i, :])
        np.savetxt("Data/PlantData.csv", artificialData, delimiter=",")
        np.savetxt("Data/PlantParams.csv", trueParamSample, delimiter=",")
