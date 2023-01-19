import diffrax as dx
import jax.numpy as jnp
import numpy as np
from jax import vmap

from epic import logger
from epic.core.model import (
    ArtificialModelInterface,
    JaxModel,
    VisualizationModelInterface,
)


class Corona(JaxModel, VisualizationModelInterface):
    def getDataBounds(self):
        return np.array([[0.0, 4.0], [0.0, 40.0], [0.0, 80.0], [0.0, 3.5]])

    def getParamBounds(self):
        return np.array([[-4.0, 0.0], [-2.0, 2.0], [-1.0, 3.0]])

    def getParamSamplingLimits(self):
        return np.array([[-4.5, 0.5], [-2.0, 3.0], [-2.0, 3.0]])

    def getCentralParam(self):
        return np.array([-1.8, 0.0, 0.7])

    @classmethod
    def forward(cls, logParam):
        param = jnp.power(10, logParam)
        xInit = jnp.array([999.0, 0.0, 1.0, 0.0])

        def rhs(t, x, param):
            return jnp.array(
                [
                    -param[0] * x[0] * x[2],
                    param[0] * x[0] * x[2] - param[1] * x[1],
                    param[1] * x[1] - param[2] * x[2],
                    param[2] * x[2],
                ]
            )

        term = dx.ODETerm(rhs)
        solver = dx.Kvaerno5()
        saveat = dx.SaveAt(ts=[0.0, 1.0, 2.0, 5.0, 15.0])
        stepsize_controller = dx.PIDController(rtol=1e-5, atol=1e-5)

        try:
            odeSol = dx.diffeqsolve(
                term,
                solver,
                t0=0.0,
                t1=15.0,
                dt0=0.1,
                y0=xInit,
                args=param,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
            )
            return odeSol.ys[1:5, 2]

        except Exception as e:
            logger.warn("ODE solution not possible!", exc_info=e)
            return np.array([-np.inf, -np.inf, -np.inf, -np.inf])


class CoronaArtificial(Corona, ArtificialModelInterface):
    def generateArtificialData(self, numSamples=1000):
        lowerBound = np.array([-1.9, -0.1, 0.6])
        upperBound = np.array([-1.7, 0.1, 0.8])

        trueParamSample = lowerBound + (
            upperBound - lowerBound
        ) * np.random.rand(numSamples, 3)

        artificialData = vmap(self.forward, in_axes=0)(trueParamSample)

        np.savetxt(
            "Data/CoronaArtificialData.csv", artificialData, delimiter=","
        )
        np.savetxt(
            "Data/CoronaArtificialParams.csv", trueParamSample, delimiter=","
        )

    def getParamSamplingLimits(self):
        return np.array([[-2.5, -1.0], [-0.75, 0.75], [0.0, 1.5]])
