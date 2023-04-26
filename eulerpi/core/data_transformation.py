from typing import Optional, Union

import jax.numpy as jnp
from jax import jit, tree_util


class DataTransformation:
    """Class for normalizing data. The data is normalized by subtracting the mean and multiplying by the inverse of the Cholesky decomposition of the covariance matrix."""

    normalizing_matrix: Union[jnp.ndarray, jnp.double]
    mean_vector: Union[jnp.ndarray, jnp.double]
    determinant: jnp.double

    def __init__(
        self,
        data: Optional[jnp.ndarray] = None,
        normalizing_matrix: Optional[jnp.ndarray] = None,
        mean_vector: Optional[jnp.ndarray] = None,
        determinant: Optional[jnp.double] = None,
    ) -> None:
        """Initialize the DataTransformation object by calculating the mean vector and the normalizing matrix.

        Args:
            data (jnp.ndarray): The data to be normalized. Columns correspond to different dimensions. Rows correspond to different observations.
        """
        if mean_vector is None:
            mean_vector = jnp.mean(data, axis=0)
        self.mean_vector = mean_vector

        if normalizing_matrix is None:
            if data.shape[1] == 1:
                normalizing_matrix = 1 / jnp.std(data)
                determinant = normalizing_matrix
            else:
                normalizing_matrix = jnp.linalg.inv(
                    jnp.linalg.cholesky(jnp.cov(data, rowvar=False))
                )  # TODO check in Silverman if this makes sense
                determinant = jnp.linalg.det(normalizing_matrix)
        self.normalizing_matrix = normalizing_matrix
        self.determinant = determinant

    @jit
    def normalize(
        self, data: Union[jnp.double, jnp.ndarray]
    ) -> Union[jnp.double, jnp.ndarray]:
        """Normalize the given data.

        Args:
            data (Union[jnp.double, jnp.ndarray]): The data to be normalized. Columns correspond to different dimensions. Rows correspond to different observations.

        Returns:
            Union[jnp.double, jnp.ndarray]: The normalized data.
        """
        if isinstance(self.normalizing_matrix, jnp.ndarray):
            if self.normalizing_matrix.ndim > 1:
                return jnp.transpose(
                    jnp.matmul(
                        self.normalizing_matrix,
                        jnp.transpose(data - self.mean_vector),
                    )
                )
        return self.normalizing_matrix * (data - self.mean_vector)

    def _tree_flatten(self):
        """This function is used by JAX to flatten the object for JIT compilation."""
        children = ()  # arrays / dynamic values
        aux_data = {
            "normalizing_matrix": self.normalizing_matrix,
            "mean_vector": self.mean_vector,
            "determinant": self.determinant,
        }  # static values

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """This function is used by JAX to unflatten the object for JIT compilation."""
        return cls(*children, **aux_data)


tree_util.register_pytree_node(
    DataTransformation,
    DataTransformation._tree_flatten,
    DataTransformation._tree_unflatten,
)
