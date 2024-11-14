from typing import Optional

import jax.numpy as jnp
from sklearn.decomposition import PCA

from .data_transformation import DataTransformation


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
    ) -> DataTransformation:
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
