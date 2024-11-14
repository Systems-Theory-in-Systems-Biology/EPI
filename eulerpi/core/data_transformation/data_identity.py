import jax.numpy as jnp

from .data_transformation import DataTransformation


class DataIdentity(DataTransformation):
    """The identity transformation. Does not change the data."""

    def __init__(
        self,
    ):
        super().__init__()

    def transform(self, data: jnp.ndarray) -> jnp.ndarray:
        """Returns the data unchanged.

        Args:
            data (jnp.ndarray): The data which should be transformed.

        Returns:
            jnp.ndarray: The data unchanged.
        """
        return data

    def jacobian(self, data: jnp.ndarray) -> jnp.ndarray:
        """Returns the identity matrix.

        Args:
            data (jnp.ndarray): The data at which the jacobian should be evaluated.

        Returns:
            jnp.ndarray: The identity matrix.
        """
        return jnp.eye(data.shape[-1])
