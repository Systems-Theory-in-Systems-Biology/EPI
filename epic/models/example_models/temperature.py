from epic.models.model import Model, ArtificialModelInterface
import jax.numpy as jnp
import numpy as np

# TODO: Does this model realyl not provide a visGrid?
class Temperature(Model):
    def forward(self, param):
        lowT = -30.0
        highT = 30.0
        res = jnp.array([lowT + (highT - lowT) * jnp.cos(jnp.abs(param[0]))])
        return res

    def getCentralParam(self):
        return np.array([np.pi / 4.0])

class TemperatureArtificial(Temperature, ArtificialModelInterface):
    def generateArtificialData(self):
        rawTrueParamSample = np.loadtxt(
        "Data/TemperatureArtificialParams.csv", delimiter=","
        )
        trueParamSample = np.zeros((rawTrueParamSample.shape[0], 1))
        trueParamSample[:, 0] = rawTrueParamSample

        artificialData = np.zeros((trueParamSample.shape[0], 1))

        for i in range(trueParamSample.shape[0]):
            artificialData[i, 0] = self.forward(trueParamSample[i, :])

        np.savetxt(
            "Data/TemperatureArtificialData.csv", artificialData, delimiter=","
        )