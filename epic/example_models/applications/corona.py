import diffrax as dx
import jax.numpy as jnp
import numpy as np
from jax import vmap

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
            print("ODE solution not possible!")
            print(repr(e))
            return np.array([-np.inf, -np.inf, -np.inf, -np.inf])


class CoronaArtificial(Corona, ArtificialModelInterface):
    def generateArtificialData(self, numSamples=1000):
        lowerBound = np.array([-1.9, -0.1, 0.6])
        upperBound = np.array([-1.7, 0.1, 0.8])

        trueParamSample = lowerBound + (
            upperBound - lowerBound
        ) * np.random.rand(numSamples, 3)

        # Options to calculate the data from the param samples
        #
        # 1. For loop: Veeeeeery slooooooow
        # artificialData = np.zeros((numSamples, 4))
        # for j in tqdm(range(numSamples)):
        #     artificialData[j, :] = self.forward(trueParamSample[j, :])
        #
        # 2. TODO: Vectorizing the forward call in the model class
        # This would mean changing our conventions for the param shape.
        # Could be done but would require more effort.
        #
        # 3. Parallelization using multiprocessing
        # See the stock model.
        # Downside: Pickling again :/. Must use global functions or other tricks to pickle...
        #
        # 4. Using vmap
        # Really low effort and fast!
        # Downside: no progressbar ;)

        # Using multiprocessing + batching would possibly be even faster. Depending whether using jax or numpy as backend.
        # jax has a single core "bug": https://github.com/google/jax/issues/5022
        # TODO: re-evaluate where to use numpy and where to use jax due to single-cpu!

        artificialData = vmap(self.forward, in_axes=0)(trueParamSample)

        np.savetxt(
            "Data/CoronaArtificialData.csv", artificialData, delimiter=","
        )
        np.savetxt(
            "Data/CoronaArtificialParams.csv", trueParamSample, delimiter=","
        )

    def getParamSamplingLimits(self):
        return np.array([[-2.5, -1.0], [-0.75, 0.75], [0.0, 1.5]])
