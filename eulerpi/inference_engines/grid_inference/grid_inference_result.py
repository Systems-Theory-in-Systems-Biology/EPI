from ..inference_result import InferenceResult
import numpy as np
from typing import Union


class GridInferenceResult(InferenceResult):

    def density(self, params: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate the parameter density at the given parameter.

        Args:
            params (np.ndarray): The parameter at which to evaluate the density.

        Returns:
            float: The parameter density at the given parameter.
        """
        pass
