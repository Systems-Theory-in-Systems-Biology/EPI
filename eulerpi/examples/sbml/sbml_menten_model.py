import importlib

import numpy as np

from eulerpi.core.model import ArtificialModelInterface, SBMLModel


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
        sbml_file = importlib.resources.path(
            "eulerpi.examples.sbml", "sbml_menten_model.xml"
        )
        timepoints = np.array([0.5, 1.0, 2.0])
        param_ids = ["Km", "kcat"]
        state_ids = ["s1"]

        super().__init__(
            sbml_file,
            timepoints,
            central_param,
            param_limits,
            param_ids,
            state_ids=state_ids,
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
