import numpy as np

from epic.core.model import ArtificialModelInterface, Model
from epic.example_models.cpp import cpp_model


class CppPlant(Model, ArtificialModelInterface):
    """A plant model which uses a c++ library with eigen3 to evaluate the forward pass and the gradient
    Param0: Water [0,1]
    Param1: Sun   [0,1]
    Data0: Size [0,2] # the more water and sun the better
    Data1: Health [0,1], to much water is not good, too much sun is not good
    Data2: Trauerfliegen :P
    """

    def forward(self, param):
        return cpp_model.forward(param)

    def jacobian(self, param):
        return cpp_model.jacobian(param)

    def getCentralParam(self) -> np.ndarray:
        return np.array([0.5, 0.5])

    def getParamSamplingLimits(self) -> np.ndarray:
        return np.array([[0.0, 1.0], [0.0, 1.0]])

    def generateArtificialData(self, numSamples=1000):
        # randomly create true parameters in [0,1]x[0,1]
        trueParamSample = np.random.rand(numSamples, 2)

        artificialData = np.zeros((trueParamSample.shape[0], 3))
        for i in range(trueParamSample.shape[0]):
            artificialData[i, :] = self.forward(trueParamSample[i, :])
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
