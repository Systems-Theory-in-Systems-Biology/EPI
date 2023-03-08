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

    """

    param_dim = 2
    data_dim = 3

    defaultParamSamplingLimits = np.array([[0, 1], [0, 1]])
    defaultcentral_param = np.array([0.5, 0.5])

    def forward(self, param):
        return cpp_model.forward(param)

    def jacobian(self, param):
        return cpp_model.jacobian(param)

    def generate_artificial_params(self, num_samples: int):
        return np.random.rand(num_samples, 2)
