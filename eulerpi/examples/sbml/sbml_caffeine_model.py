from importlib.resources import as_file, files

import numpy as np

from eulerpi.models import ArtificialModelInterface, SBMLModel


class CaffeineSBMLModel(SBMLModel, ArtificialModelInterface):

    CENTRAL_PARAM = np.array([1.0, 1.0])
    PARAM_LIMITS = np.array([[0.0, 2.0], [0.0, 2.0]])

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        **kwargs,
    ) -> None:
        param_ids = ["A", "B"]
        timepoints = np.array([0.5, 1.0])
        sbml_files = files("eulerpi.examples.sbml")
        sbml_file_name = "Caffeine_2Wks_Exponential_decay.xml"
        with as_file(sbml_files.joinpath(sbml_file_name)) as sbml_file:
            super().__init__(
                sbml_file,
                central_param,
                param_limits,
                timepoints,
                param_ids,
                **kwargs,
            )

    def generate_artificial_params(self, num_samples: int) -> np.ndarray:
        diff0 = 0.2
        diff1 = 0.2
        params = np.random.rand(num_samples, self.param_dim)
        params[:, 0] *= diff0
        params[:, 0] += 1.0 - diff0 / 2.0

        params[:, 1] *= diff1
        params[:, 1] += 1.0 - diff1 / 2.0
        return params
