from typing import Tuple

import jax.numpy as jnp
from jax import jit, tree_util

from .affine_transformation import AffineTransformation


@jit
def compute_normalization(data) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mean_vector = jnp.mean(data, axis=0)
    cov = jnp.cov(data, rowvar=False)
    L = jnp.linalg.cholesky(jnp.atleast_2d(cov))
    normalizing_matrix = jnp.linalg.inv(L)
    shift_vector = -normalizing_matrix @ mean_vector

    # Use jnp.squeeze to reduce dimensions if normalizing_matrix is (1, 1)
    normalizing_matrix = jnp.squeeze(normalizing_matrix)

    return normalizing_matrix, shift_vector


class DataNormalization(AffineTransformation):
    """Class for normalizing data. The data is normalized by subtracting the mean and multiplying by the inverse of the Cholesky decomposition of the covariance matrix."""

    def __init__(self, data: jnp.ndarray):
        """Initialize a DataNormalization object.

        Args:
            data (jnp.ndarray): The data from which to calculate the mean vector and normalizing matrix.
        """
        normalizing_matrix, shift_vector = compute_normalization(data)

        super().__init__(normalizing_matrix, shift_vector)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """Unflatten the DataNormalization object for JAX."""
        # Create an instance of DataNormalization without invoking its __init__
        instance = cls.__new__(cls)  # Bypasses DataNormalization.__init__
        # Initialize instance using AffineTransformation's __init__
        AffineTransformation.__init__(
            instance, *children
        )  # Calls AffineTransformation's __init__
        return instance  # Return the correctly initialized DataNormalization instance


# Register the pytree node for JAX to handle serialization for DataNormalization
tree_util.register_pytree_node(
    DataNormalization,
    DataNormalization._tree_flatten,
    DataNormalization._tree_unflatten,
)
