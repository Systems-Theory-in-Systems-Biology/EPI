import numpy as np


class InferenceResult:

    def __init__(
        self,
        params: np.ndarray,
        pushforward_evals: np.ndarray,
        density_evals: np.ndarray,
        meta_data: dict,
    ):
        self.params = params
        self.pushforward_evals = pushforward_evals
        self.density_evals = density_evals
        self.meta_data = meta_data

    def density(self, param: np.ndarray, **kwargs) -> float:
        raise NotImplementedError(
            "The density evaluation has to be implemented in the subclass."
        )

    def save(self, data_storage):
        pass

    @classmethod
    def load(cls, data_storage):
        pass
