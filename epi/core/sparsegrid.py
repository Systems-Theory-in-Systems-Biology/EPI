"""This module provides functions to handle create Sparse Grids (SGs_) and work with them.
   All pure SG functions are defined on the unit hypercube $[0,1]^d$.


.. _SGs: https://en.wikipedia.org/wiki/Sparse_grid
"""

import numpy as np


def basis1D(
    points1D: np.ndarray, centre1D: np.double, level: int
) -> np.ndarray:

    """Evaluate a 1D hat function in an array of doubles. The hat is centered around centre1D
       and the hat's level defines its support. The support shrinks exponentially with growing level and a level of 0 is equivalent with full support on [0,1].

    Input: points1D (np.1darray of 1D evaluation coordinate doubles)
           centre1D (np.double indicating the centre of the hat within the interval [0,1])
           level (int specifying the size/extend/support of the hat function)

    Output: (np.1darray (size equivalent to size of points1D) of hat function evaluations)

    """

    return np.maximum(
        0, 1 - np.power(2, level + 1) * np.abs(points1D - centre1D)
    )


def basisnD(
    points: np.ndarray, centre: np.ndarray, levels: np.ndarray
) -> np.ndarray:

    """Use a tensor product to generalise the 1D basis function to arbitrarily high dimensions.

    Input: points (np.2darray of shape #Points x #Dims indicating the basis evaluation coordinates in nD)
           centre (np.1darray of shape #Dims defining the nD centre of an nD basis function)
           levels (np.1darray of type int and shape #Dims defining one basis function level per dimension)

    Output: basisEval (np.1darray of shape #Points returning one nD basis evaluation per specified evaluation point)
    """

    # Initialize the basis evaluation of each point as 1
    basisEval = np.ones(points.shape[0])

    # loop over all dimensions
    for d in range(points.shape[1]):
        # Multipy the current basis evaluation with the evaluation result of the current dimension
        basisEval *= basis1D(points[:, d], centre[d], levels[d])

    return basisEval


def meshgrid2Matrix(meshgrid: list) -> np.ndarray:
    """Convert a np.meshgrid into a np.2darray of grid points.
    The function is mainly used when assigning grid points to Smolnyak-Subspaces.

    Input: meshgrid (list of np.arrays returned by np.meshgrid)

    Output: matrix (np.2darray of shape #Points x #Dims defining all grid points)
    """

    # calculate the shape of the matrix and initialize with 0s
    dim = len(meshgrid)
    nPoints = np.prod(meshgrid[0].shape)

    matrix = np.zeros((nPoints, dim))

    # read out the respective meshgrid entry for each matrix entry
    for d in range(dim):
        linearMeshSlice = np.reshape(meshgrid[d], -1)
        for p in range(nPoints):
            matrix[p, d] = linearMeshSlice[p]

    return matrix
