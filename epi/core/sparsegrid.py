"""This module provides functions to handle create Sparse Grids (SGs_) and work with them.
   All pure SG functions are defined on the unit hypercube $[0,1]^d$.


.. _SGs: https://en.wikipedia.org/wiki/Sparse_grid
"""

import typing
from functools import partial
from multiprocessing import Pool

import numpy as np

from epi.core.kde import calcKernelWidth
from epi.core.model import Model
from epi.core.transformations import evalLogTransformedDensity

NUM_LEVELS = 5
NUM_PROCESSES = 4


def basis1D(
    points1D: np.ndarray, centre1D: np.double, level: int
) -> np.ndarray:

    """Evaluate a 1D hat function in an array of doubles. The hat is centered around centre1D
    and the hat's level defines its support. The support shrinks exponentially with growing level and a level of 0 is equivalent with full support on [0,1].

    Args:
        points1D(np.ndarray): The points at which the hat function should be evaluated.
        centre1D(np.double): The centre of the hat function.
        level(int): The level of the hat function. The level defines the support of the hat function.

    Returns:
        np.ndarray: The hat function evaluated at the given points.

    """

    return np.maximum(
        0, 1 - np.power(2, level + 1) * np.abs(points1D - centre1D)
    )


def basisnD(
    points: np.ndarray, centre: np.ndarray, levels: np.ndarray
) -> np.ndarray:

    """Use a tensor product to generalise the 1D basis function to arbitrarily high dimensions.

    Args:
        points(np.ndarray): The points at which the basis function should be evaluated. Shape: (numPoints, numDims)
        centre(np.ndarray): The centre of the basis function. Shape: (numDims,)
        levels(np.ndarray): The levels of the basis function. Shape: (numDims,)

    Returns:
        np.ndarray: The basis function evaluated at the given points. Shape: (numPoints,)

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

    Args:
        meshgrid(list): A list of np.arrays returned by np.meshgrid

    Returns:
        np.ndarray: A matrix of shape #Points x #Dims defining all grid points

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


class SparseGrid(object):
    """Each object of this class respresents a sparse grid.
    In this implementation, a sparse grid is a list of Smolnyak-subspaces.
    Each subspace is in principle a regular grid of a certain grid width but every second grid point is negelcted.

    Attributes:
        dim (int): The dimension of the sparse grid. This is the same as the dimension of the parameter space.
        maxLevelSum (int): The maximum sum of all levels of the subspaces. This is the same as the maximum level of the sparse grid.
        subspaceList (list): A list of all subspaces that are part of the sparse grid.
        levels2index (dict): A dictionary that maps the level combination of a subspace to its index in the subspaceList.
        nSubspaces (int): The number of subspaces in the sparse grid.
        nPoints (int): The number of grid points in the sparse grid.
        indexList4TopDownSparseGridTraverse (list): A list of indices that defines an ordering of subspaces where low-level subspaces come before high-level ones.
        allPoints (np.ndarray): A matrix of shape #Points x #Dims defining all grid points in the sparse grid.

    """

    def __init__(self, dim: int, maxLevelSum: int) -> None:
        """Constructor for a sparse grid.
        A sparse grid is uniquely defined by its dimension and a level sum that must not be exceeded by any of the Smolnyak subspaces.
        A subspace's levels define how fine the grid is resolved in each of the respective dimensions.
        The position of a certain subspace within the list of subspaces can be tracked using the levels2index dictionary.
        As we only limit the sum of all levels, the sparse grids implemented here are not refined in a dimension-dependent way.

        Args:
            dim (int): The dimension of the sparse grid. This is the same as the dimension of the parameter space.
            maxLevelSum (int): The maximum sum of all levels of the subspaces. This is the same as the maximum level of the sparse grid.

        """

        self.dim = dim
        self.maxLevelSum = maxLevelSum

        # initiation of the root, list of subspaces and dictionary that maps the level-combination to the list-index
        root = Subspace(np.zeros(dim, dtype="int"), self)
        self.subspaceList = [root]
        self.levels2index = {}
        self.levels2index[tuple(np.zeros(dim, dtype="int"))] = 0

        # refine root by calling the recursive function refineSubspace and count resulting subspaces and grid points
        self.refineSubspace(np.zeros(dim, dtype="int"), 0)
        self.nSubspaces = len(self.subspaceList)
        self.computeNPoints()

        # create an ordering of subspaces where low-level subspaces come before high-level ones
        self.computeIndexList4TopDownSparseGridTraverse()

        # collect all points from all subspaces
        self.computeAllPoints()

    def refineSubspace(
        self, currentLevels: np.ndarray, indexRefinedLevel: int
    ) -> None:
        """Recursive function used to accumulate all subspaces up to a specified level sum in the form of a list
        It returns the list itself together with a dictionary that maps the level-combination of each subspace onto its index inside the list.
        This function only lists each subspace once.

        Args:
            currentLevels (np.ndarray): The level combination of the subspace that is currently being refined. Shape (dim,)
            indexRefinedLevel (int): The index of the level that was refined to form the current subspace.

        """

        # This loop makes sure that each subspace is only counted once.
        # Achieved by storing the index that got altered to form the current subspace and letting the current
        # ... subspace only refine level indices with similar or higher entry number in the levels array.
        for i in range(indexRefinedLevel, self.dim):

            # derive the level increment array and calculate new level
            levelsIncrement = np.zeros(self.dim, dtype="int")
            levelsIncrement[i] = 1

            newLevels = currentLevels + levelsIncrement

            # kill-condition for recursion if max level is reached
            if np.sum(newLevels) <= self.maxLevelSum:

                # store refined subspace in list and dictionary
                self.levels2index[tuple(newLevels)] = len(self.subspaceList)
                self.subspaceList.append(Subspace(newLevels, self))

                # recursive call to refine refined subspace
                self.refineSubspace(newLevels, i)

    def computeNPoints(self):
        """Iterates over all subspaces of the sparse grid and accumulates the total number of gridpoints."""

        # initiate the counter to be 0
        self.nPoints = 0

        # loop over all subspaces
        for s in range(self.nSubspaces):

            # get current subspace
            currentSubspace = self.subspaceList[s]
            # add the number of points in the current subspace
            self.nPoints += currentSubspace.nPoints

    def computeAllPoints(self):
        """Collect all SG points in one array by iterating over all subspaces."""

        # allocate enough storage for all points
        self.points = np.zeros((self.nPoints, self.dim))

        # initiate a counter for the number of already counted points
        numIncludedPoints = 0

        # loop over all subspaces of the SG
        for i in range(self.nSubspaces):
            # traverse the SG in a top-down manner
            currentSubspace = self.subspaceList[
                self.indexList4TopDownSparseGridTraverse[i]
            ]

            # copy the points from the subspace into the array of the SG
            self.points[
                numIncludedPoints : numIncludedPoints
                + currentSubspace.nPoints,
                :,
            ] = currentSubspace.points

            # increase the counter accordingly
            numIncludedPoints += currentSubspace.nPoints

    def computeIndexList4TopDownSparseGridTraverse(self):
        """Create an ordering of subspaces where low-level subspaces come before high-level ones."""

        # allocate storage to count the sum of levels of each subspace
        levelSums = np.zeros(self.nSubspaces, dtype="int")

        # loop over all subspaces and sum over their levels array
        for i in range(self.nSubspaces):
            levelSums[i] = np.sum(list(self.levels2index)[i])

        # argument sort by the just-calculated level-sum
        self.indexList4TopDownSparseGridTraverse = np.argsort(levelSums)

    # TODO: Shouldn't an eval function return something?
    def evalFunctionSG(self, function: typing.Callable):
        """Evaluate the provided function for all subspaces of a sparse grid by using Subspace.evalFunction

        Args:
            function (typing.Callable): The function that is to be evaluated. It must be possible to evaluate the function in a single sparse grid point.
        """

        # loop over all subspaces
        for s in range(self.nSubspaces):
            # call evalFunction for the current subspace
            self.subspaceList[s].evalFunction(function)

    def computeCoefficients(self):
        """When using sparse grids for function interpolation (and quadrature),
        this function computes the coefficients of all basis function of the whole sparse grid.

        Args:

        Returns:

        """

        # loop over all smolnyak subspaces in a low to high level order
        for s in range(self.nSubspaces):
            currentSubspace = self.subspaceList[
                self.indexList4TopDownSparseGridTraverse[s]
            ]

            # calculate coefficients for the current subspace (consider contributions from "larger" basis functions)
            currentSubspace.coeffs = (
                currentSubspace.fEval - currentSubspace.lowerLevelContributions
            )

            # pass up contributions arising from the just-computed coefficients to
            # ... all higher levels if there are any
            if np.sum(currentSubspace.levels) < self.maxLevelSum:
                currentSubspace.passContributions2HigherLevels()

    def computeIntegral(self):
        """Perform sparse grid integration over whole Sparse Grid using the computed coefficients (coeffs) and the volume of each basis function (basisFuncVol)"""
        # initialise the integral to be 0
        self.integral = 0

        # loop over all subspaces
        for s in range(self.nSubspaces):
            # exrtact the current subspace
            currentSubspace = self.subspaceList[s]

            # multiply the volume of each basis function with the sum of all coefficients of this subspace and add the result to the integral
            # (this implicitely uses that all basis functions of a given subspace have the same volume)
            self.integral += (
                np.sum(currentSubspace.coeffs) * currentSubspace.basisFuncVol
            )


class Subspace(object):
    """Objects represent one Smolnyak-Subspace of a sparse grid and are only defined by a level for each dimension."""

    def __init__(self, levels: np.ndarray, SG: SparseGrid) -> None:
        """Initialize the subspace by assigning a level, dimension, number of points and the actual points themselves.

        Args:
            levels (np.ndarray): The level of the subspace in each dimension. Shape: (#Dims, )
            SG (SparseGrid): The sparse grid of which the current subspace is a part.

        """

        # fill all known information into the class variables
        self.SG = SG
        self.levels = levels
        self.dim = levels.shape[0]
        self.basisFuncVol = np.power(0.5, np.sum(self.levels + 1))
        self.nPoints = np.prod(np.power(2, levels))

        # this variable is created without being directly filled -> Caution when using it; Check for reliable data
        self.lowerLevelContributions = np.zeros(self.nPoints)

        # Create all points of the current subspace and fill self.points
        # Start by creating an empty list to store the coordinates of all single dimensions
        singleDimPoints = []

        # loop over all dimensions
        for d in range(self.dim):

            # append a list of 1d coordinates for each dimension
            singleDimPoints.append(
                np.linspace(
                    1 / np.power(2, levels[d] + 1),
                    1 - 1 / np.power(2, levels[d] + 1),
                    np.power(2, levels[d]),
                )
            )

        # create all possible combinations from the 1d coordinate arrays
        meshgrid = np.meshgrid(*singleDimPoints)

        # convert the numpy meshgrid to a matrix of all points with shape (nPoints,dim)
        self.points = meshgrid2Matrix(meshgrid)

    def evalFunction(self, function: typing.Callable):
        """Evaluate a function in all points of the respective subspace.
        This function is typically called by SparseGrid.evalFunctionSG.

        Args:
            function (typing.Callable): The function that is to be evaluated. It must be possible to evaluate the function in a single sparse grid point.

        """
        # create an empty array of size #Points
        self.fEval = np.zeros(self.nPoints)

        # loop over all grid points of the subspace
        for i in range(self.nPoints):
            # evaluate the provided function in the current grid point
            self.fEval[i] = function(self.points[i, :])

    def passContributions2HigherLevels(self):
        """During sparse grid interpolation, this function passes contributions to all subspaces with higher level."""

        # loop over all subspaces of the SG (this can be made more efficient)
        for s in range(self.SG.nSubspaces):
            higherLevelSubspace = self.SG.subspaceList[s]

            # check if the higherLevelSubspace indeed has a higher level
            if np.sum(higherLevelSubspace.levels) > np.sum(self.levels):

                # loop over all points in the mother subspace and add contributions to lower levels
                for p in range(self.nPoints):
                    higherLevelSubspace.lowerLevelContributions += (
                        basisnD(
                            higherLevelSubspace.points,
                            self.points[p, :],
                            self.levels,
                        )
                        * self.coeffs[p]
                    )


# TODO: Move to inference? And include slices and correct result paths with result_manager
def sparseGridInference(
    model: Model,
    data: np.ndarray,
    numLevels: int = NUM_LEVELS,
    numProcesses: int = NUM_PROCESSES,
):
    """Evaluates the transformed parameter density over a set of points resembling a sparse grid, thereby attempting parameter inference. If a data path is given, it is used to load the data for the model. Else, the default data path of the model is used.

    Args:
      model(Model): The model describing the mapping from parameters to data.
      data(np.ndarray): The data to be used for inference.
      numLevels(int): Maximum sparse grid level depth that mainly defines the number of points. Defaults to NUM_LEVELS.
      numProcesses(int): number of processes to use for parallel evaluation of the model. Defaults to NUM_PROCESSES.

    """

    # Load data, data standard deviations and model characteristics for the specified model.
    dataDim = model.dataDim
    dataStdevs = calcKernelWidth(data)
    paramDim = model.paramDim

    # build the sparse grid over [0,1]^paramDim
    grid = SparseGrid(paramDim, numLevels)

    # get the model's parameter limits
    parameterLimits = model.paramLimits

    # scale the sparse grid points from [0,1]^paramDim to the scaled parameter space
    scaledSparseGridPoints = parameterLimits[:, 0] + grid.points * (
        parameterLimits[:, 1] - parameterLimits[:, 0]
    )

    # allocate Memory for the parameters, their simulation evaluation and their probability density
    samplerResults = np.zeros((grid.nPoints, paramDim + model.dataDim + 1))

    # Create a pool of worker processes
    pool = Pool(processes=numProcesses)

    # evaluate the probability density transformation for all sparse grid points in parallel
    parResults = pool.map(
        partial(
            evalLogTransformedDensity,
            model=model,
            data=data,
            dataStdevs=dataStdevs,
        ),
        scaledSparseGridPoints,
    )

    # close the worker pool
    pool.close()
    pool.join()

    # extract the parameter, simulation result and transformed density evaluation
    for i in range(grid.nPoints):
        samplerResults[i, :] = parResults[i][1]

    # Save all sparse grid evaluation results in separate .csv files that also indicate the sparse grid level.
    np.savetxt(
        "Applications/"
        + model.name
        + "/Params/SG"
        + str(numLevels)
        + "Levels.csv",
        samplerResults[:, 0:paramDim],
        delimiter=",",
    )
    np.savetxt(
        "Applications/"
        + model.name
        + "/SimResults/SG"
        + str(numLevels)
        + "Levels.csv",
        samplerResults[:, paramDim : paramDim + dataDim],
        delimiter=",",
    )
    np.savetxt(
        "Applications/"
        + model.name
        + "/DensityEvals/SG"
        + str(numLevels)
        + "Levels.csv",
        samplerResults[:, -1],
        delimiter=",",
    )


# TODO: Use maybe only one function for storing samplerResults
# TODO: Plotting for general dimension
