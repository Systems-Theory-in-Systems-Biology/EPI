from typing import Union

import jax.numpy as jnp
from jax import jit, tree_util

from .data_transformation import DataTransformation


class AffineTransformation(DataTransformation):
    """Class for applying an affine data transformation, y=Ax+b"""

    def __init__(self, A: jnp.ndarray, b: jnp.ndarray):
        """Initialize a AffineTransformation obkect.

        Args:
            A (jnp.ndarray): The matrix representing the linear part of the transformation
            b (jnp.ndarray): The vector shifting the data
        """
        super().__init__()

        self.A = A
        self.b = b

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
        return jnp.inner(data + self.b, self.A)

    def jacobian(self, data: jnp.ndarray) -> jnp.ndarray:
        return self.A

    def _tree_flatten(self):
        """This function is used by JAX to flatten the object for JIT compilation."""
        children = (
            self.A,
            self.b,
        )  # arrays / dynamic values
        aux_data = {}  # static values

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """This function is used by JAX to unflatten the object for JIT compilation."""
        return cls(*children, **aux_data)


tree_util.register_pytree_node(
    AffineTransformation,
    AffineTransformation._tree_flatten,
    AffineTransformation._tree_unflatten,
)
