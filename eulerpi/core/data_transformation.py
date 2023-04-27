from abc import ABC, abstractmethod
from typing import Optional, Union

import jax.numpy as jnp
from jax import jit, tree_util
from sklearn.decomposition import PCA


class DataTransformation(ABC):
    """Class for normalizing data. The data is normalized by subtracting the mean and multiplying by the inverse of the Cholesky decomposition of the covariance matrix."""

    @abstractmethod
    def transform(self, data: jnp.ndarray) -> jnp.ndarray:
        """Transform the given data."""
        raise NotImplementedError()

    @abstractmethod
    def jacobian(self, data: jnp.ndarray) -> jnp.ndarray:
        """Return the jacobian of the transformation at the given data point(s)."""
        raise NotImplementedError()


class DataIdentity(DataTransformation):
    """The identity transformation. Does not change the data."""

    def __init__(
        self,
    ):
        super().__init__()

    def transform(self, data: jnp.ndarray) -> jnp.ndarray:
        return data

    def jacobian(self, data: jnp.ndarray) -> jnp.double:
        return jnp.eye(data.shape[1])


class DataNormalizer(DataTransformation):
    """Class for normalizing data. The data is normalized by subtracting the mean and multiplying by the inverse of the Cholesky decomposition of the covariance matrix."""

    def __init__(
        self,
        normalizing_matrix: Optional[jnp.ndarray] = None,
        mean_vector: Optional[jnp.ndarray] = None,
        determinant: Optional[jnp.double] = None,
    ):
        super().__init__()

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
        aux_data = {
            "determinant": self.determinant,
        }  # static values

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """This function is used by JAX to unflatten the object for JIT compilation."""
        return cls(*children, **aux_data)


tree_util.register_pytree_node(
    DataNormalizer,
    DataNormalizer._tree_flatten,
    DataNormalizer._tree_unflatten,
)


class DataPCA(DataTransformation):
    """The DataPCA class can be used to transform the data using the Principal Component Analysis."""

    def __init__(
        self,
        pca: Optional[PCA] = None,
    ):
        super().__init__()

        self.pca = pca

    @classmethod
    def from_data(
        cls, data: jnp.ndarray, n_components: Optional[int] = None
    ) -> "DataTransformation":
        """Initialize a DataTransformation object by calculating the mean vector, the normalizing matrix and the determinant from the given data."""
        instance = cls()
        instance.pca = PCA(n_components=n_components)
        instance.pca.fit(data)
        return instance

    def transform(self, data: jnp.ndarray) -> jnp.ndarray:
        return self.pca.transform(data)

    def jacobian(self, data: jnp.ndarray) -> jnp.ndarray:
        return self.pca.components_.T
