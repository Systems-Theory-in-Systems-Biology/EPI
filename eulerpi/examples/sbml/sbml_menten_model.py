from importlib.resources import as_file, files

import numpy as np

from eulerpi.core.models import ArtificialModelInterface, SBMLModel


class MentenSBMLModel(SBMLModel, ArtificialModelInterface):
    """The MentenModel is a simple model for the Michaelis-Menten enzyme kinetics.
    It requires a few adaptations to the SBMLModel class to work with the inference, because
    the first output of the model does not depend on the parameters and the second output is linear dependent on the third output.
    See data_dim, forward, jacobian, forward_and_jacobian.
    """

    CENTRAL_PARAM = np.array([50.0, 1.0])
    PARAM_LIMITS = np.array([[0.0, 100.0], [0.0, 2.0]])

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        **kwargs,
    ) -> None:
        timepoints = np.array([0.5, 1.0])
        param_ids = ["Km", "kcat"]
        state_ids = ["s1"]
        sbml_files = files("eulerpi.examples.sbml")
        sbml_file_name = "sbml_menten_model.xml"
        with as_file(sbml_files.joinpath(sbml_file_name)) as sbml_file:
            super().__init__(
                sbml_file,
                central_param,
                param_limits,
                timepoints,
                param_ids,
                state_ids,
                **kwargs,
            )

    def generate_artificial_params(self, num_samples: int) -> np.ndarray:
        diff0 = 5.0
        diff1 = 0.2
        params = np.random.rand(num_samples, self.param_dim)
        params[:, 0] *= diff0
        params[:, 0] += 50.0 - diff0 / 2.0

        params[:, 1] *= diff1
        params[:, 1] += 1.0 - diff1 / 2.0
        return params
