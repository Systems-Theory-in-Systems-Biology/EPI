from enum import Enum


class DataTransformationType(Enum):
    """Defines the allowed :py:mod:`DataTransformation<eulerpi.core.data_transformation>` types for the inference function."""

    Identity = 0  #: Do not transform the data
    Normalize = 1  #: Normalize the data to zero mean and no correlation
    PCA = 2  #: Apply a pca transformation
    Custom = 3  #: Apply a custom transformation to the data
