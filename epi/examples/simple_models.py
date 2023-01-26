import jax.numpy as jnp
import numpy as np
from jax import vmap

from epi.core.model import (
    ArtificialModelInterface,
    JaxModel,
    VisualizationModelInterface,
)


class Linear(JaxModel, ArtificialModelInterface, VisualizationModelInterface):
    @classmethod
    def forward(cls, param):
        return jnp.array([param[0] * 10, (-2.0) * param[1] - 2.0])

    def getCentralParam(self):
        return np.array([0.5, 0.5])

    def getParamSamplingLimits(self):
        return np.array([[-10.0, 11.0], [-10.0, 11.0]])

    def generateArtificialData(
        self, numSamples=ArtificialModelInterface.NUM_ARTIFICIAL_SAMPLES
    ):

        # randomly create true parameters in [0,1]^2
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

    def getParamBounds(self, scale=1.0) -> np.ndarray:
        return np.array([[-0.2, 1.2], [-0.2, 1.2]])

    def getDataBounds(self, scale=1.0) -> np.ndarray:
        return np.array([[-2.0, 12.0], [-4.4, -1.6]])


class Exponential(JaxModel, VisualizationModelInterface):
    @classmethod
    def forward(cls, param):
        return jnp.array([param[0] * jnp.exp(1), jnp.exp(param[1])])

    def getCentralParam(self) -> np.ndarray:
        return np.array([1.0, 1.0])

    def getParamSamplingLimits(self) -> np.ndarray:
        return np.array([[1.0, 2.0], [1.0, 2.0]])

    def getParamBounds(self) -> np.ndarray:
        return np.array([0.8, 2.2], [0.8, 2.2])

    # TODO: ???
    # KDExGrid = np.linspace(0.8 * np.exp(1), 2.2 * np.exp(1), KDEresolution)
    # KDEyGrid = np.linspace(np.exp(0.8), np.exp(2.2), KDEresolution)
    # KDExMesh, KDEyMesh = np.meshgrid(KDExGrid, KDEyGrid)
    def getDataBounds(self) -> np.ndarray:
        return self.forward(self.getParamBounds())


class LinearODE(JaxModel, ArtificialModelInterface):
    @classmethod
    def forward(cls, param):
        return jnp.array(
            [
                param[0] * jnp.exp(param[1] * 1.0),
                param[0] * jnp.exp(param[1] * 2.0),
            ]
        )

    def getParamSamplingLimits(self):
        return np.array([[-10.0, 23.0], [-10.0, 23.0]])

    def getCentralParam(self):
        return np.array([1.5, 1.5])

    def generateArtificialData(self, numSamples=1000):
        # randomly create true parameters in [1,2]^2
        trueParamSample = np.random.rand(numSamples, 2) + 1

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
