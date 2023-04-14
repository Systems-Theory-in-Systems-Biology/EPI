from enum import Enum


class DenseGridType(Enum):
    """The type of grid to be used."""

    EQUIDISTANT = 0  #: The equidistant grid has the same distance between two grid points in each dimension.
    CHEBYSHEV = 1  #: The Chebyshev grid is a tensor product of Chebyshev polynomial roots. They are optimal for polynomial interpolation and quadrature.
