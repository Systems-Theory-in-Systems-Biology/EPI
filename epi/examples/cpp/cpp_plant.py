import numpy as np

from epi.core.model import ArtificialModelInterface, Model
from epi.examples.cpp import cpp_model


class CppPlant(Model, ArtificialModelInterface):
    """A plant model which uses a c++ library with eigen3 to evaluate the forward pass and the gradient
    Param0: Water [0,1]
    Param1: Sun   [0,1]
    Data0: Size [0,2] # the more water and sun the better
    Data1: Health [0,1], to much water is not good, too much sun is not good
    Data2: Sciarid ;)

    Args:

    Returns:

    """

    paramDim = 2
    dataDim = 3

    defaultParamSamplingLimits = np.array([[0, 1], [0, 1]])
    defaultCentralParam = np.array([0.5, 0.5])

    def forward(self, param):
        """

        Args:
          param:

        Returns:

        """
        return cpp_model.forward(param)

    def jacobian(self, param):
        """

        Args:
          param:

        Returns:

        """
        return cpp_model.jacobian(param)

    def generateArtificialParams(self, numSamples: int):
        """

        Args:
          numSamples: int:

        Returns:

        """
        return np.random.rand(numSamples, 2)
