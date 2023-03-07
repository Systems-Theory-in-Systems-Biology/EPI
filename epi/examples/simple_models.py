import jax.numpy as jnp
import numpy as np

from epi.core.model import ArtificialModelInterface, JaxModel


class Linear(JaxModel, ArtificialModelInterface):
    paramDim = 2
    dataDim = 2

    defaultParamSamplingLimits = np.array([[-10.0, 11.0], [-10.0, 11.0]])
    defaultCentralParam = np.array([0.5, 0.5])

    @classmethod
    def forward(cls, param):
        return jnp.array([param[0] * 10, (-2.0) * param[1] - 2.0])

    def generateArtificialParams(self, numSamples: int):
        return np.random.rand(numSamples, self.paramDim)

    def getParamBounds(self, scale=1.0) -> np.ndarray:
        return np.array([[-0.2, 1.2], [-0.2, 1.2]])

    def getDataBounds(self, scale=1.0) -> np.ndarray:
        return np.array([[-2.0, 12.0], [-4.4, -1.6]])


class Exponential(JaxModel):
    paramDim = 2
    dataDim = 2

    defaultParamSamplingLimits = np.array([[1.0, 2.0], [1.0, 2.0]])
    defaultCentralParam = np.array([1.0, 1.0])

    @classmethod
    def forward(cls, param):
        return jnp.array([param[0] * jnp.exp(1), jnp.exp(param[1])])

    def getParamBounds(self) -> np.ndarray:
        return np.array([0.8, 2.2], [0.8, 2.2])

    # TODO: ???
    # KDExGrid = np.linspace(0.8 * np.exp(1), 2.2 * np.exp(1), KDEresolution)
    # KDEyGrid = np.linspace(np.exp(0.8), np.exp(2.2), KDEresolution)
    # KDExMesh, KDEyMesh = np.meshgrid(KDExGrid, KDEyGrid)
    def getDataBounds(self) -> np.ndarray:
        return self.forward(self.getParamBounds())


class LinearODE(JaxModel, ArtificialModelInterface):

    paramDim = 2
    dataDim = 2

    defaultParamSamplingLimits = np.array([[-10.0, 23.0], [-10.0, 23.0]])
    defaultCentralParam = np.array([1.5, 1.5])

    @classmethod
    def forward(cls, param):
        return jnp.array(
            [
                param[0] * jnp.exp(param[1] * 1.0),
                param[0] * jnp.exp(param[1] * 2.0),
            ]
        )

    def generateArtificialParams(self, numSamples: int):
        return np.random.rand(numSamples, 2) + 1
