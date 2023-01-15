from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit

from epic.core.model import ArtificialModelInterface, Model


class Temperature(Model):
    def __init__(self, delete: bool = False, create: bool = True) -> None:
        super().__init__(delete, create)
        self.lowT = -30.0
        self.highT = 30.0

    # TODO: Provide a class which uses functool.partial for forward function and puts self.arg in it?
    # Of course to use jitting. partial(jit, static_argnums=[0,2,])
    # Maybe not the temperature model because it is used in the tutorial ...
    @partial(jit, static_argnums=0)
    def forward(self, param):
        return self.calc_forward(param, self.highT, self.lowT)

    def calc_forward(self, param, highT, lowT):
        res = jnp.array([lowT + (highT - lowT) * jnp.cos(jnp.abs(param[0]))])
        return res

    def __hash__(self):
        return hash((self.lowT, self.highT))

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
        trueParamSample = rawTrueParamSample[..., np.newaxis]

        artificialData = np.zeros((trueParamSample.shape[0], 1))

        for i in range(trueParamSample.shape[0]):
            artificialData[i, 0] = self.forward(trueParamSample[i, :])

        np.savetxt(
            "Data/TemperatureArtificialData.csv", artificialData, delimiter=","
        )
