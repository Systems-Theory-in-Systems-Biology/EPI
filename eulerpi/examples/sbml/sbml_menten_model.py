import importlib

import numpy as np

from eulerpi.core.model import ArtificialModelInterface, SBMLModel


class MentenSBMLModel(SBMLModel, ArtificialModelInterface):

    CENTRAL_PARAM = np.array([50.0, 1.0])
    PARAM_LIMITS = np.array([[0.0, 100.0], [0.0, 2.0]])

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        **kwargs,
    ) -> None:
        sbml_file = importlib.resources.path(
            "eulerpi.examples.sbml", "sbml_menten_model.xml"
        )
        param_names = ["Km", "kcat"]
        super().__init__(
            sbml_file,
            central_param,
            param_limits,
            param_names,
            **kwargs,
        )

    # Overwrite the data_dim, forward, jacobian, and forward_and_jacobian methods to remove the first variable which is not dependent on the parameters
    @property
    def data_dim(self) -> int:
        return 2

    def forward(self, params) -> np.ndarray:
        return super().forward(params)[1:]

    def jacobian(self, params) -> np.ndarray:
        return super().jacobian(params)[1:, :]

    def forward_and_jacobian(self, params) -> np.ndarray:
        val, jac = super().forward_and_jacobian(params)
        return val[1:], jac[1:, :]

    def generate_artificial_params(self, num_samples: int) -> np.ndarray:
        diff0 = 5.0
        diff1 = 0.2
        params = np.random.rand(num_samples, self.param_dim)
        params[:, 0] *= diff0
        params[:, 0] += 50.0 - diff0 / 2.0

        params[:, 1] *= diff1
        params[:, 1] += 1.0 - diff1 / 2.0
        return params
