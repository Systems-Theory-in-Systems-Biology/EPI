from abc import ABC, abstractmethod

import jax.numpy as jnp


class DensityEstimator(ABC):
    @abstractmethod
    def __call__(data_point: jnp.ndarray) -> jnp.double:
        raise NotImplementedError(
            "Every DensityEstimator for eulerpi needs to implement the __call__ function"
        )
