from typing import Callable, Union

import numpy as np
from numpy.polynomial.chebyshev import chebpts1


class Grid:
    def __init__(self, grid_data: np.ndarray):
        """
        Initializes the grid with the provided grid data.

        Args:
            grid_data (np.ndarray): The grid data (points) as [n_points,dim] array
        """
        self.grid_data = grid_data


class ChebyshevGrid(Grid):
    def __init__(self, limits, num_grid_points):
        """ "Generate a grid with the given number of grid points for each dimension.

        Args:
            num_grid_points(np.ndarray): The number of grid points for each dimension.
            limits(np.ndarray): The limits for each dimension.
        """
        flatten = True
        mesh = generate_chebyshev_grid(num_grid_points, limits, flatten)

        super().__init__(mesh)


class RegularGrid(Grid):
    def __init__(self, limits, num_grid_points):
        """ "Generate a grid with the given number of grid points for each dimension.

        Args:
            num_grid_points(np.ndarray): The number of grid points for each dimension.
            limits(np.ndarray): The limits for each dimension.
        """
        flatten = True
        mesh = generate_regular_grid(num_grid_points, limits, flatten)

        super().__init__(mesh)


class GridEvaluator:
    def __init__(self, grid: Grid, function: Callable):
        self.grid = grid
        self.function = function


def generate_regular_grid(
    num_grid_points: np.ndarray,
    limits: np.ndarray,
    flatten=False,
) -> Union[np.ndarray, list[np.ndarray]]:
    """Generate a grid with the given number of grid points for each dimension.

    Args:
        num_grid_points(np.ndarray): The number of grid points for each dimension.
        limits(np.ndarray): The limits for each dimension.
        flatten(bool): If True, the grid is returned as a flatten array. If False, the grid is returned as a list of arrays, one for each dimension. (Default value = False)

    Returns:
        np.ndarray: The grid containing the grid points.

    """
    ndim = num_grid_points.size
    axes = [
        np.linspace(limits[i][0], limits[i][1], num=num_grid_points[i])
        for i in range(ndim)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    if flatten:
        return np.array(mesh).reshape(ndim, -1).T
    else:
        return mesh


def generate_chebyshev_grid(
    num_grid_points: np.ndarray,
    limits: np.ndarray,
    flatten=False,
) -> Union[np.ndarray, list[np.ndarray]]:
    """Generate a grid with the given number of grid points for each dimension.

    Args:
        num_grid_points(np.ndarray): The number of grid points for each dimension.
        limits(np.ndarray): The limits for each dimension.
        flatten(bool): If True, the grid is returned as a flatten array. If False, the grid is returned as a list of arrays, one for each dimension. (Default value = False)

    Returns:
        np.ndarray: The grid containing the grid points.

    """
    ndim = num_grid_points.size
    axes = [
        chebpts1(num_grid_points[i]) * (limits[i][1] - limits[i][0]) / 2
        + (limits[i][1] + limits[i][0]) / 2
        for i in range(ndim)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    if flatten:
        return np.array(mesh).reshape(ndim, -1).T
    else:
        return mesh
