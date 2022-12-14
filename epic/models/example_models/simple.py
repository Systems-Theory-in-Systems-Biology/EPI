import jax.numpy as jnp
import numpy as np

from epic.models.model import ArtificialModelInterface, Model


class Linear(Model, ArtificialModelInterface):
    def forward(self, param):
        return jnp.array([param[0] * 10, (-2.0) * param[1] - 2.0])

    def getCentralParam(self):
        return np.array([0.5, 0.5])

    def getParamSamplingLimits(self):
        return np.array([[-10.0, 11.0], [-10.0, 11.0]])

    def generateArtificialData(self):
        numSamples = 1000

        # randomly create true parameters in [0,1]^2
        trueParamSample = np.random.rand(numSamples, 2)

        artificialData = np.zeros((trueParamSample.shape[0], 2))

        for i in range(trueParamSample.shape[0]):
            artificialData[i, :] = self.forward(trueParamSample[i, :])

        np.savetxt("Data/LinearData.csv", artificialData, delimiter=",")
        np.savetxt("Data/LinearParams.csv", trueParamSample, delimiter=",")


# TODO: Complete this class by providing getCentralParam(self)
class Exponential(Model):
    def forward(self, param):
        return jnp.array([param[0] * jnp.exp(1), jnp.exp(param[1])])


class LinearODE(Model, ArtificialModelInterface):
    def forward(self, param):
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

    def generateArtificialData(self):
        numSamples = 1000

        # randomly create true parameters in [1,2]^2
        trueParamSample = np.random.rand(numSamples, 2) + 1

        artificialData = np.zeros((trueParamSample.shape[0], 2))

        for i in range(trueParamSample.shape[0]):
            artificialData[i, :] = self.forward(trueParamSample[i, :])

        np.savetxt("Data/LinearODEData.csv", artificialData, delimiter=",")
        np.savetxt("Data/LinearODEParams.csv", trueParamSample, delimiter=",")
