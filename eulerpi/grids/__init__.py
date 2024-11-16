"""This package provides a grid base class which is used in eulerpi and multiple grid implementations for use in the inference function. You can register your own class using the `register_grid` class decorator."""

from .chebyshev_grid import ChebyshevGrid

# Dont remove the imports. The imports cause the grids to register themselves in the grid factory register
from .equidistant_grid import EquidistantGrid
from .sparse_grid import SparseGrid
