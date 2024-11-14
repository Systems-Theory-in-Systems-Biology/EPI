from abc import ABC, abstractmethod

import jax.numpy as jnp


class DataTransformation(ABC):
    """Abstract base class for all data transformations

    Data transformations can be used to improve the performance of the :py:func:`inference <eulerpi.core.inference.inference>` function
    by improving the quality of the :py:mod:`kernel density estimate <eulerpi.core.evaluation.kde>`.
    """

    @abstractmethod
    def transform(self, data: jnp.ndarray) -> jnp.ndarray:
        """Transform the given data point(s)

        Args:
            data (jnp.ndarray): The data to be transformed. Columns correspond to different dimensions. Rows correspond to different observations.

        Raises:
            NotImplementedError: Raised if the transform is not implemented in the subclass.

        Returns:
            jnp.ndarray: The transformed data point(s).
        """
        raise NotImplementedError()

    @abstractmethod
    def jacobian(self, data: jnp.ndarray) -> jnp.ndarray:
        """Calculate the jacobian of the transformation at the given data point(s).

        Args:
            data (jnp.ndarray): The data at which the jacobian should be evaluated.

        Raises:
            NotImplementedError: Raised if the jacobian is not implemented in the subclass.

        Returns:
            jnp.ndarray: The jacobian of the transformation at the given data point(s).
        """
        raise NotImplementedError()
