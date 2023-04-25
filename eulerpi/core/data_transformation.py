from typing import Union

import numpy as np


class DataTransformation:
    """Class for normalizing data. The data is normalized by subtracting the mean and multiplying by the inverse of the Cholesky decomposition of the covariance matrix."""

    normalizing_matrix: np.ndarray
    mean_vector: np.ndarray
    determinant: float

    def __init__(self, data: np.ndarray) -> None:
        """Initialize the DataTransformation object by calculating the mean vector and the normalizing matrix.

        Args:
            data (np.ndarray): The data to be normalized. Columns correspond to different dimensions. Rows correspond to different observations.
        """
        self.mean_vector = np.mean(data, axis=0)
        if data.shape[1] == 1:
            self.normalizing_matrix = 1 / np.std(data)
            self.determinant = self.normalizing_matrix
        else:
            self.normalizing_matrix = np.linalg.inv(
                np.linalg.cholesky(np.cov(data, rowvar=False))
            )  # TODO check in Silverman if this makes sense
            self.determinant = np.linalg.det(self.normalizing_matrix)

    def normalize(self, data: Union[float, np.ndarray]) -> np.ndarray:
        """Normalize the given data.

        Args:
            data (Union[float, np.ndarray]): The data to be normalized. Columns correspond to different dimensions. Rows correspond to different observations.

        Returns:
            np.ndarray: The normalized data.
        """
        if data is np.ndarray:
            return np.matmul(data - self.mean_vector, self.normalizing_matrix)
        return (data - self.mean_vector) * self.normalizing_matrix
