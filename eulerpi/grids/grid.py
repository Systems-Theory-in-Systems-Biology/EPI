from abc import ABC, abstractmethod

import numpy as np


class Grid(ABC):
    def __init__(self, limits):
        self.limits = np.atleast_2d(limits)

    @property
    @abstractmethod
    def grid_points(self) -> np.ndarray:
        """Return a flat array of shape (n,dim) of all grid points

        Returns:
            np.ndarray: The points of the grid
        """
        raise NotImplementedError(
            "The abstract grid base class has no grid points"
        )