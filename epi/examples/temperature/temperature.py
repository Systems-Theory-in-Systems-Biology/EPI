import importlib

import jax.numpy as jnp
import numpy as np

from epi.core.model import ArtificialModelInterface, Model

# from functools import partial
# from jax import jit


class Temperature(Model):

    paramDim = 1
    dataDim = 1

    defaultParamSamplingLimits = np.array([[0, np.pi / 2]])
    defaultCentralParam = np.array([np.pi / 4.0])

    def __init__(self, name: str = None) -> None:
        super().__init__(name=name)

        self.dataPath = importlib.resources.path(
            "epi.examples.temperature", "TemperatureData.csv"
        )

    def forward(self, param):
        lowT = -30.0
        highT = 30.0
        res = jnp.array([lowT + (highT - lowT) * jnp.cos(jnp.abs(param[0]))])
        return res

    def jacobian(self, param):
        return jnp.array([60.0 * jnp.sin(jnp.abs(param[0]))])


class TemperatureArtificial(Temperature, ArtificialModelInterface):
    def generateArtificialParams(self, numSamples: int):
        paramPath = importlib.resources.path(
            "epi.examples.temperature", "TemperatureArtificialParams.csv"
        )
        trueParamSample = np.loadtxt(paramPath, delimiter=",", ndmin=2)
        return trueParamSample


class TemperatureWithFixedParams(Temperature):
    def __init__(self, name: str = None) -> None:
        super().__init__(name=name)
        self.lowT = -30.0
        self.highT = 30.0

    def forward(self, param):
        return self.calcForward(param, self.highT, self.lowT)

    def calcForward(self, param, highT, lowT):
        res = jnp.array([lowT + (highT - lowT) * jnp.cos(jnp.abs(param[0]))])
        return res

    def jacobian(self, param):
        return jnp.array(
            [(self.highT - self.lowT) * jnp.sin(jnp.abs(param[0]))]
        )
