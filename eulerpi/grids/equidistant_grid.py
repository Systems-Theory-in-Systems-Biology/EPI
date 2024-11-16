from typing import Union

import numpy as np

from .grid import Grid
from .grid_factory import register_grid


def generate_equidistant_grid(
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


@register_grid("EQUIDISTANT")
class EquidistantGrid(Grid):
    """Grid with equal spacing"""

    def __init__(self, limits, num_grid_points):
        """ "Generate a grid with the given number of grid points for each dimension.

        Args:
            num_grid_points(np.ndarray): The number of grid points for each dimension.
            limits(np.ndarray): The limits for each dimension.
        """
        if isinstance(num_grid_points, int):
            num_grid_points = (
                np.ones((limits.shape[0]), dtype=int) * num_grid_points
            )
        limits = np.atleast_2d(limits)
        super().__init__(limits, num_grid_points)
        flatten = True
        self.mesh = generate_equidistant_grid(num_grid_points, limits, flatten)

    @property
    def grid_points(self):
        return self.mesh
