from typing import Optional, Union

import numpy as np


class DataTransformation:
    """Class for normalizing data. The data is normalized by subtracting the mean and multiplying by the inverse of the Cholesky decomposition of the covariance matrix."""

    normalizing_matrix: Union[np.ndarray, float]
    mean_vector: Union[np.ndarray, float]
    determinant: float

    def __init__(self, data: Optional[np.ndarray] = None) -> None:
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

    def normalize(
        self, data: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Normalize the given data.

        Args:
            data (Union[float, np.ndarray]): The data to be normalized. Columns correspond to different dimensions. Rows correspond to different observations.

        Returns:
            Union[float, np.ndarray]: The normalized data.
        """
        if isinstance(self.normalizing_matrix, np.ndarray):
            if self.normalizing_matrix.ndim > 1:
                return np.transpose(
                    np.matmul(
                        self.normalizing_matrix,
                        np.transpose(data - self.mean_vector),
                    )
                )
        return self.normalizing_matrix * (data - self.mean_vector)
