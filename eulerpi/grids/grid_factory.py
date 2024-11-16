"""Factory and registry for grid construction.

This module defines a registry of available grid types and provides a factory
function for constructing grid objects. It comes with built in support for equidistant, Chebyshev,
and sparse grids.

Grid types:
- EQUIDISTANT: A regular grid with equidistant points.
- CHEBYSHEV: A grid based on Chebyshev polynomial roots, suitable for interpolation.
- SPARSE: A sparse grid for joint distribution evaluation (experimental).

Example:
    grid = construct_grid("EQUIDISTANT", limits=np.array([[0, 1], [0, 1]]), levels_or_points=10)
"""

from typing import Dict, Type, Union

import numpy as np

from .grid import Grid

#: Registry for grid types available for the grid based inference
GRID_REGISTRY: Dict[str, Type[Grid]] = {}


def register_grid(name: str):
    def decorator(cls: Type[Grid]):
        GRID_REGISTRY[name.upper()] = cls
        return cls

    return decorator


def construct_grid(
    grid_type: str,
    limits: np.ndarray,
    levels_or_points: Union[int, np.ndarray],
) -> Grid:
    """Construct a grid of the specified type.

    Args:
        grid_type (str): The type of grid
        limits (np.ndarray): The grid limits, shape (dim, 2).
        levels_or_points (Union[int, np.ndarray]): Levels or points based on grid type.

    Returns:
        Grid: An instance of the requested grid type.

    Raises:
        ValueError: If the grid type is not recognized.
    """
    if grid_type not in GRID_REGISTRY:
        raise ValueError(
            f"Unknown grid type: {grid_type}. Available types: {list(GRID_REGISTRY.keys())}"
        )
    GridClass = GRID_REGISTRY[grid_type]
    return GridClass(limits, levels_or_points)
