import jax.numpy as jnp
import numpy as np

from epic.core.model import ArtificialModelInterface, Model


# TODO: Does this model realyl not provide a visGrid?
class Temperature(Model):
    def __init__(self) -> None:
        super().__init__()
        self.lowT = -30.0
        self.highT = 30.0

    def forward(self, param):
        return jnp.array(
            [
                Model.lowT
                + (self.highT - self.lowT) * jnp.cos(jnp.abs(param[0]))
            ]
        )

    def jacobian(self, param):
        return jnp.array(
            [(self.highT - self.lowT) * jnp.sin(jnp.abs(param[0]))]
        )

    def getCentralParam(self):
        return np.array([np.pi / 4.0])

    def getParamSamplingLimits(self):
        return np.array([[0, np.pi / 2]])


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
