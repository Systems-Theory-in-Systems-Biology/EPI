from abc import ABC, abstractmethod
from typing import Optional, Union

import jax.numpy as jnp
from jax import jit, tree_util
from sklearn.decomposition import PCA


class DataTransformation(ABC):
    """Abstract base class for all data transformations

    Data transformations can be used to improve the performance of the :py:func:`inference <eulerpi.core.inference.inference>` function
    by improving the quality of the :py:mod:`kernel density estimate <eulerpi.core.kde>`.
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


class DataNormalizer(DataTransformation):
    """Class for normalizing data. The data is normalized by subtracting the mean and multiplying by the inverse of the Cholesky decomposition of the covariance matrix."""

    def __init__(
        self,
        normalizing_matrix: Optional[jnp.ndarray] = None,
        mean_vector: Optional[jnp.ndarray] = None,
    ):
        """Initialize a DataNormalizer object.

        Args:
            normalizing_matrix (Optional[jnp.ndarray], optional): The normalizing matrix. Defaults to None.
            mean_vector (Optional[jnp.ndarray], optional): The mean / shift vector. Defaults to None.
        """
        super().__init__()

        self.normalizing_matrix = normalizing_matrix
        self.mean_vector = mean_vector

    @classmethod
    def from_data(cls, data: jnp.ndarray) -> "DataTransformation":
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
        """Initialize a DataPCA object.

        Args:
            pca (Optional[PCA], optional): The PCA object to be used for the transformation. Defaults to None.
        """
        super().__init__()

        self.pca = pca

    @classmethod
    def from_data(
        cls, data: jnp.ndarray, n_components: Optional[int] = None
    ) -> "DataTransformation":
        """Initialize a DataPCA object by calculating the PCA from the given data.

        Args:
            data (jnp.ndarray): The data to be used for the PCA.
            n_components (Optional[int], optional): The number of components to keep. If None is passed, min(n_samples,n_features) is used. Defaults to None.

        Returns:
            DataTransformation: The initialized DataPCA object.
        """
        instance = cls()
        instance.pca = PCA(n_components=n_components)
        instance.pca.fit(data)
        return instance

    def transform(self, data: jnp.ndarray) -> jnp.ndarray:
        """Transform the given data using the PCA.

        Args:
            data (jnp.ndarray): The data to be transformed.

        Returns:
            jnp.ndarray: The data projected onto and expressed in the basis of the principal components.
        """
        result = self.pca.transform(data.reshape(-1, data.shape[-1])).reshape(
            -1, self.pca.n_components_
        )

        # if the input data was 1D, the output should be 1D as well
        if data.ndim == 1:
            result = result.flatten()

        return result

    def jacobian(self, data: jnp.ndarray) -> jnp.ndarray:
        """Return the jacobian of the pca transformation at the given data point(s).

        Args:
            data (jnp.ndarray): The data point(s) at which the jacobian should be evaluated.

        Returns:
            jnp.ndarray: The jacobian of the pca transformation at the given data point(s).
        """
        return self.pca.components_
