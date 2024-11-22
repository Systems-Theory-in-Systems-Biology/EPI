import numpy as np

from eulerpi.examples.cpp import cpp_model
from eulerpi.models import ArtificialModelInterface, BaseModel


class CppPlant(BaseModel, ArtificialModelInterface):
    """A plant model which uses a c++ library with eigen3 to evaluate the forward pass and the gradient
    Param0: Water [0,1]
    Param1: Sun   [0,1]
    Data0: Size [0,2] # the more water and sun the better
    Data1: Health [0,1], to much water is not good, too much sun is not good
    Data2: Sciarid ;)

    """

    param_dim = 2
    data_dim = 3

    PARAM_LIMITS = np.array([[0, 1], [0, 1]])
    CENTRAL_PARAM = np.array([0.5, 0.5])

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: str = None,
        **kwargs,
    ) -> None:
        super().__init__(
            central_param,
            param_limits,
            name,
            **kwargs,
        )

    def forward(self, param):
        return cpp_model.forward(param)

    def jacobian(self, param):
        return cpp_model.jacobian(param)

    def generate_artificial_params(self, num_samples: int):
        return np.random.rand(num_samples, 2)
