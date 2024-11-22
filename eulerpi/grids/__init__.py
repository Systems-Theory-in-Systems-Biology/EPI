"""This package provides a grid base class which is used in eulerpi and multiple grid implementations for use in the inference function. You can register your own class using the `register_grid` class decorator."""

from .chebyshev_grid import ChebyshevGrid
from .equidistant_grid import EquidistantGrid
from .sparse_grid import SparseGrid
