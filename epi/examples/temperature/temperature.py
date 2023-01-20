import importlib

import jax.numpy as jnp
import numpy as np
from jax import vmap

from epi.core.model import ArtificialModelInterface, Model

# from functools import partial
# from jax import jit


class Temperature(Model):
    def __init__(self, delete: bool = False, create: bool = True) -> None:
        super().__init__(delete, create)
        self.lowT = -30.0
        self.highT = 30.0

        self.data_path = importlib.resources.path(
            "epi.examples.temperature", "TemperatureData.csv"
        )

    # TODO: Provide a class which uses functool.partial for forward function and puts self.arg in it?
    # Of course to use jitting. partial(jit, static_argnums=[0,2,])
    # Maybe not the temperature model because it is used in the tutorial ...
    # @partial(jit, static_argnums=0) # this slows down the code?!
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
        paramPath = importlib.resources.path(
            "epi.examples.temperature", "TemperatureArtificialParams.csv"
        )
        rawTrueParamSample = np.loadtxt(paramPath, delimiter=",")
        trueParamSample = rawTrueParamSample[..., np.newaxis]

        artificialData = vmap(self.forward, in_axes=0)(trueParamSample)

        np.savetxt(
            f"Data/{self.getModelName()}Data.csv",
            artificialData,
            delimiter=",",
        )
