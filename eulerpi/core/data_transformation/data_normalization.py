from typing import Optional, Union

import jax.numpy as jnp
from jax import jit, tree_util

from .data_transformation import DataTransformation


class DataNormalization(DataTransformation):
    """Class for normalizing data. The data is normalized by subtracting the mean and multiplying by the inverse of the Cholesky decomposition of the covariance matrix."""

    def __init__(
        self,
        normalizing_matrix: Optional[jnp.ndarray] = None,
        mean_vector: Optional[jnp.ndarray] = None,
    ):
        """Initialize a DataNormalization object.

        Args:
            normalizing_matrix (Optional[jnp.ndarray], optional): The normalizing matrix. Defaults to None.
            mean_vector (Optional[jnp.ndarray], optional): The mean / shift vector. Defaults to None.
        """
        super().__init__()

        self.normalizing_matrix = normalizing_matrix
        self.mean_vector = mean_vector

    @classmethod
    def from_data(cls, data: jnp.ndarray) -> DataTransformation:
        """Initialize a DataTransformation object by calculating the mean vector and normalizing matrix from the given data.

        Args:
            data (jnp.ndarray): The data from which to calculate the mean vector and normalizing matrix.

        Returns:
            DataTransformation: The initialized DataTransformation object.
        """
        instance = cls()
        instance.mean_vector = jnp.mean(data, axis=0)
        cov = jnp.cov(data, rowvar=False)
        L = jnp.linalg.cholesky(jnp.atleast_2d(cov))
        instance.normalizing_matrix = jnp.linalg.inv(L)

        if instance.normalizing_matrix.shape == (1, 1):
            instance.normalizing_matrix = instance.normalizing_matrix[0, 0]

        return instance

    @classmethod
    def from_transformation(
        cls,
        mean_vector: jnp.ndarray,
        normalizing_matrix: jnp.ndarray,
    ) -> "DataTransformation":
        """Initialize a DataTransformation object from the given mean vector, normalizing matrix and determinant.

        Args:
            mean_vector (jnp.ndarray): The vector to shift the data by.
            normalizing_matrix (jnp.ndarray): The matrix to multiply the data by.

        Returns:
            DataTransformation: The initialized DataTransformation object.
        """
        instance = cls()
        instance.mean_vector = mean_vector
        instance.normalizing_matrix = normalizing_matrix
        return instance

    @jit
    def transform(
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

    def jacobian(self, data: jnp.ndarray) -> jnp.ndarray:
        return self.normalizing_matrix

    def _tree_flatten(self):
        """This function is used by JAX to flatten the object for JIT compilation."""
        children = (
            self.normalizing_matrix,
            self.mean_vector,
        )  # arrays / dynamic values
        aux_data = {}  # static values

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """This function is used by JAX to unflatten the object for JIT compilation."""
        return cls(*children, **aux_data)


tree_util.register_pytree_node(
    DataNormalization,
    DataNormalization._tree_flatten,
    DataNormalization._tree_unflatten,
)
