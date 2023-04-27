from typing import Optional, Union

import jax.numpy as jnp
from jax import jit, tree_util


class DataTransformation:
    """Class for normalizing data. The data is normalized by subtracting the mean and multiplying by the inverse of the Cholesky decomposition of the covariance matrix."""

    def __init__(
        self,
        normalizing_matrix: Optional[jnp.ndarray] = None,
        mean_vector: Optional[jnp.ndarray] = None,
        determinant: Optional[jnp.double] = None,
    ):
        self.normalizing_matrix = normalizing_matrix
        self.mean_vector = mean_vector
        self.determinant = determinant

    @classmethod
    def from_data(cls, data: jnp.ndarray) -> "DataTransformation":
        """Initialize a DataTransformation object by calculating the mean vector, the normalizing matrix and the determinant from the given data."""
        instance = cls()
        instance.mean_vector = jnp.mean(data, axis=0)
        cov = jnp.cov(data, rowvar=False)
        L = jnp.linalg.cholesky(jnp.atleast_2d(cov))
        instance.normalizing_matrix = jnp.linalg.inv(L)
        instance.determinant = jnp.linalg.det(instance.normalizing_matrix)
        return instance

    @classmethod
    def from_transformation(
        cls,
        mean_vector: jnp.ndarray,
        normalizing_matrix: jnp.ndarray,
        determinant: Optional[jnp.double] = None,
    ) -> "DataTransformation":
        """Initialize a DataTransformation object from the given mean vector, normalizing matrix and determinant."""
        instance = cls()
        instance.mean_vector = mean_vector
        instance.normalizing_matrix = normalizing_matrix
        if determinant is not None:
            instance.determinant = determinant
        else:
            instance.determinant = jnp.linalg.det(normalizing_matrix)
        return instance

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
        # possible shapes
        # normalizing matrix: (d, d)
        # data: (n, d)
        # data: (d)
        # The correct output shape is (n, d) or (d) depending on the input shape.
        return jnp.inner(data - self.mean_vector, self.normalizing_matrix)

    def _tree_flatten(self):
        """This function is used by JAX to flatten the object for JIT compilation."""
        children = (
            self.normalizing_matrix,
            self.mean_vector,
        )  # arrays / dynamic values
        aux_data = {
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
