"""Data transformations can be used to improve the performance of the :py:func:`inference <eulerpi.inference.inference>` function by improving the quality of the kernel density estimate.

This subpackage contains all predefined data transformations and an abstract base class for custom data transformations.
"""

from .affine_transformation import AffineTransformation as AffineTransformation
from .data_identity import DataIdentity as DataIdentity
from .data_normalization import DataNormalization as DataNormalization
from .data_pca import DataPCA as DataPCA
from .data_transformation import DataTransformation as DataTransformation
