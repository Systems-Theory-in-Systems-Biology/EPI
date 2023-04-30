from enum import Enum


class DataTransformationType(Enum):
    """Defines the allowed DataTransformation types for the inference function."""

    Identity = 0  # No data transformation
    Normalize = 1  # Normalizes the data to zero mean and no correlation
    PCA = 2  # Apply a pca transformation
    Custom = 3  # Apply a custom transformation to the data
