import importlib
from typing import Optional

import numpy as np

from epi.core.model import ArtificialModelInterface, SBMLModel


class MentenSBMLModel(SBMLModel, ArtificialModelInterface):

    defaultcentral_param = np.array([50.0, 1.0])
    defaultParamSamplingLimits = np.array([[0.0, 100.0], [0.0, 2.0]])

    def __init__(
        self,
        central_param: Optional[np.ndarray] = None,
        param_limits: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ) -> None:
        sbml_file = importlib.resources.path(
            "epi.examples.sbml", "sbml_menten_model.xml"
        )
        param_names = ["Km", "kcat"]
        super().__init__(
            sbml_file,
            param_names,
            1.0,
            False,
            central_param,
            param_limits,
            name,
        )

    # Overwrite the forward, jacobian, and valjac methods to remove the first variable which is not dependent on the parameters
    def forward(self, params) -> np.ndarray:
        return super().forward(params)[1:]

    def jacobian(self, params) -> np.ndarray:
        return super().jacobian(params)[1:, :]

    def valjac(self, params) -> np.ndarray:
        val, jac = super().valjac(params)
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
