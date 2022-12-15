import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit

from epic.core.model import ArtificialModelInterface, Model


def autodiff(cls):
    cls.initFwAndBw()
    return cls


@autodiff
class AutodiffModel(Model, ArtificialModelInterface):
    @classmethod
    def my_fw(cls, param):
        return jnp.array(
            [
                param[0] * param[1],
                jnp.prod(jnp.sin(jnp.pi * param)),
                jnp.exp(param[0]) - 0.999,
            ]
        )

    @classmethod
    def initFwAndBw(cls):
        cls.fw = jit(cls.my_fw)
        cls.bw = jit(jacrev(cls.fw))

    def getModelName(self) -> str:
        return "Plant"

    def forward(self, param):
        return type(self).fw(param)

    def jacobian(self, param):
        return type(self).bw(param)

    def getCentralParam(self) -> np.ndarray:
        return np.array([0.5, 0.5])

    def getParamSamplingLimits(self) -> np.ndarray:
        return np.array([[0.0, 1.0], [0.0, 1.0]])

    def generateArtificialData(self):
        numSamples = 1000
        # randomly create true parameters in [0,1]x[0,1]
        trueParamSample = np.random.rand(numSamples, 2)

        artificialData = np.zeros((trueParamSample.shape[0], 3))
        for i in range(trueParamSample.shape[0]):
            artificialData[i, :] = self.forward(trueParamSample[i, :])
        np.savetxt("Data/PlantData.csv", artificialData, delimiter=",")
        np.savetxt("Data/PlantParams.csv", trueParamSample, delimiter=",")
