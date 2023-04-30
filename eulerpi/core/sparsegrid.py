""" This module provides functions to handle create Sparse Grids (SGs_) and work with them.
    All pure SG functions are defined on the unit hypercube $[0,1]^d$.

    .. warning::

        The inference with this class is not tested and not recommended for use!

.. _SGs: https://en.wikipedia.org/wiki/Sparse_grid
"""

import typing
from itertools import repeat
from multiprocessing import get_context

import numpy as np

from eulerpi import logger
from eulerpi.core.data_transformation import DataTransformation
from eulerpi.core.inference_types import InferenceType
from eulerpi.core.kde import calc_kernel_width
from eulerpi.core.model import Model
from eulerpi.core.result_manager import ResultManager
from eulerpi.core.transformations import evaluate_density


def basis_1d(
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


def basis_nd(
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
        basisEval *= basis_1d(points[:, d], centre[d], levels[d])

    return basisEval


def meshgrid2matrix(meshgrid: list) -> np.ndarray:
    """Convert a np.meshgrid into a np.2darray of grid points.
    The function is mainly used when assigning grid points to Smolnyak-Subspaces.

    Args:
        meshgrid(list): A list of np.arrays returned by np.meshgrid

    Returns:
        np.ndarray: A matrix of shape #Points x #Dims defining all grid points

    """

    # calculate the shape of the matrix and initialize with 0s
    dim = len(meshgrid)
    n_points = np.prod(meshgrid[0].shape)

    matrix = np.zeros((n_points, dim))

    # read out the respective meshgrid entry for each matrix entry
    for d in range(dim):
        linear_mesh_slice = np.reshape(meshgrid[d], -1)
        for p in range(n_points):
            matrix[p, d] = linear_mesh_slice[p]

    return matrix


class SparseGrid(object):
    """Each object of this class respresents a sparse grid.
    In this implementation, a sparse grid is a list of Smolnyak-subspaces.
    Each subspace is in principle a regular grid of a certain grid width but every second grid point is negelcted.

    Attributes:
        dim (int): The dimension of the sparse grid. This is the same as the dimension of the parameter space.
        max_level_sum (int): The maximum sum of all levels of the subspaces. This is the same as the maximum level of the sparse grid.
        subspace_list (list): A list of all subspaces that are part of the sparse grid.
        levels2index (dict): A dictionary that maps the level combination of a subspace to its index in the subspace_list.
        nSubspaces (int): The number of subspaces in the sparse grid.
        n_points (int): The number of grid points in the sparse grid.
        index_list4top_down_sparse_grid_traverse[ (list): A list of indices that defines an ordering of subspaces where low-level subspaces come before high-level ones.
        allPoints (np.ndarray): A matrix of shape #Points x #Dims defining all grid points in the sparse grid.

    """

    def __init__(self, dim: int, max_level_sum: int) -> None:
        """Constructor for a sparse grid.
        A sparse grid is uniquely defined by its dimension and a level sum that must not be exceeded by any of the Smolnyak subspaces.
        A subspace's levels define how fine the grid is resolved in each of the respective dimensions.
        The position of a certain subspace within the list of subspaces can be tracked using the levels2index dictionary.
        As we only limit the sum of all levels, the sparse grids implemented here are not refined in a dimension-dependent way.

        Args:
            dim (int): The dimension of the sparse grid. This is the same as the dimension of the parameter space.
            max_level_sum (int): The maximum sum of all levels of the subspaces. This is the same as the maximum level of the sparse grid.

        """

        self.dim = dim
        self.max_level_sum = max_level_sum

        # initiation of the root, list of subspaces and dictionary that maps the level-combination to the list-index
        root = Subspace(np.zeros(dim, dtype="int"), self)
        self.subspace_list = [root]
        self.levels2index = {}
        self.levels2index[tuple(np.zeros(dim, dtype="int"))] = 0

        # refine root by calling the recursive function refine_subspace and count resulting subspaces and grid points
        self.refine_subspace(np.zeros(dim, dtype="int"), 0)
        self.nSubspaces = len(self.subspace_list)
        self.compute_n_points()

        # create an ordering of subspaces where low-level subspaces come before high-level ones
        self.compute_index_list4top_down_sparse_grid_traverse()

        # collect all points from all subspaces
        self.compute_all_points()

    def refine_subspace(
        self, current_levels: np.ndarray, indexRefinedLevel: int
    ) -> None:
        """Recursive function used to accumulate all subspaces up to a specified level sum in the form of a list
        It returns the list itself together with a dictionary that maps the level-combination of each subspace onto its index inside the list.
        This function only lists each subspace once.

        Args:
            current_levels (np.ndarray): The level combination of the subspace that is currently being refined. Shape (dim,)
            indexRefinedLevel (int): The index of the level that was refined to form the current subspace.

        """

        # This loop makes sure that each subspace is only counted once.
        # Achieved by storing the index that got altered to form the current subspace and letting the current
        # ... subspace only refine level indices with similar or higher entry number in the levels array.
        for i in range(indexRefinedLevel, self.dim):
            # derive the level increment array and calculate new level
            levels_increment = np.zeros(self.dim, dtype="int")
            levels_increment[i] = 1

            new_levels = current_levels + levels_increment

            # kill-condition for recursion if max level is reached
            if np.sum(new_levels) <= self.max_level_sum:
                # store refined subspace in list and dictionary
                self.levels2index[tuple(new_levels)] = len(self.subspace_list)
                self.subspace_list.append(Subspace(new_levels, self))

                # recursive call to refine refined subspace
                self.refine_subspace(new_levels, i)

    def compute_n_points(self):
        """Iterates over all subspaces of the sparse grid and accumulates the total number of gridpoints."""

        # initiate the counter to be 0
        self.n_points = 0

        # loop over all subspaces
        for s in range(self.nSubspaces):
            # get current subspace
            current_subspace = self.subspace_list[s]
            # add the number of points in the current subspace
            self.n_points += current_subspace.n_points

    def compute_all_points(self):
        """Collect all SG points in one array by iterating over all subspaces."""

        # allocate enough storage for all points
        self.points = np.zeros((self.n_points, self.dim))

        # initiate a counter for the number of already counted points
        num_included_points = 0

        # loop over all subspaces of the SG
        for i in range(self.nSubspaces):
            # traverse the SG in a top-down manner
            current_subspace = self.subspace_list[
                self.index_list4top_down_sparse_grid_traverse[i]
            ]

            # copy the points from the subspace into the array of the SG
            self.points[
                num_included_points : num_included_points
                + current_subspace.n_points,
                :,
            ] = current_subspace.points

            # increase the counter accordingly
            num_included_points += current_subspace.n_points

    def compute_index_list4top_down_sparse_grid_traverse(self):
        """Create an ordering of subspaces where low-level subspaces come before high-level ones."""

        # allocate storage to count the sum of levels of each subspace
        level_sums = np.zeros(self.nSubspaces, dtype="int")

        # loop over all subspaces and sum over their levels array
        for i in range(self.nSubspaces):
            level_sums[i] = np.sum(list(self.levels2index)[i])

        # argument sort by the just-calculated level-sum
        self.index_list4top_down_sparse_grid_traverse = np.argsort(level_sums)

    # TODO: Shouldn't an eval function return something?
    def eval_function_sg(self, function: typing.Callable):
        """Evaluate the provided function for all subspaces of a sparse grid by using Subspace.eval_function

        Args:
            function (typing.Callable): The function that is to be evaluated. It must be possible to evaluate the function in a single sparse grid point.
        """

        # loop over all subspaces
        for s in range(self.nSubspaces):
            # call eval_function for the current subspace
            self.subspace_list[s].eval_function(function)

    def compute_coefficients(self):
        """When using sparse grids for function interpolation (and quadrature),
        this function computes the coefficients of all basis function of the whole sparse grid.

        Args:

        Returns:

        """

        # loop over all smolnyak subspaces in a low to high level order
        for s in range(self.nSubspaces):
            current_subspace = self.subspace_list[
                self.index_list4top_down_sparse_grid_traverse[s]
            ]

            # calculate coefficients for the current subspace (consider contributions from "larger" basis functions)
            current_subspace.coeffs = (
                current_subspace.f_eval
                - current_subspace.lower_level_contributions
            )

            # pass up contributions arising from the just-computed coefficients to
            # ... all higher levels if there are any
            if np.sum(current_subspace.levels) < self.max_level_sum:
                current_subspace.pass_contributions2higher_levels()

    def compute_integral(self):
        """Perform sparse grid integration over whole Sparse Grid using the computed coefficients (coeffs) and the volume of each basis function (basis_func_vol)"""
        # initialise the integral to be 0
        self.integral = 0

        # loop over all subspaces
        for s in range(self.nSubspaces):
            # exrtact the current subspace
            current_subspace = self.subspace_list[s]

            # multiply the volume of each basis function with the sum of all coefficients of this subspace and add the result to the integral
            # (this implicitely uses that all basis functions of a given subspace have the same volume)
            self.integral += (
                np.sum(current_subspace.coeffs)
                * current_subspace.basis_func_vol
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
        self.basis_func_vol = np.power(0.5, np.sum(self.levels + 1))
        self.n_points = np.prod(np.power(2, levels))

        # this variable is created without being directly filled -> Caution when using it; Check for reliable data
        self.lower_level_contributions = np.zeros(self.n_points)

        # Create all points of the current subspace and fill self.points
        # Start by creating an empty list to store the coordinates of all single dimensions
        single_dim_points = []

        # loop over all dimensions
        for d in range(self.dim):
            # append a list of 1d coordinates for each dimension
            single_dim_points.append(
                np.linspace(
                    1 / np.power(2, levels[d] + 1),
                    1 - 1 / np.power(2, levels[d] + 1),
                    np.power(2, levels[d]),
                )
            )

        # create all possible combinations from the 1d coordinate arrays
        meshgrid = np.meshgrid(*single_dim_points)

        # convert the numpy meshgrid to a matrix of all points with shape (n_points,dim)
        self.points = meshgrid2matrix(meshgrid)

    def eval_function(self, function: typing.Callable):
        """Evaluate a function in all points of the respective subspace.
        This function is typically called by SparseGrid.eval_function_sg.

        Args:
            function (typing.Callable): The function that is to be evaluated. It must be possible to evaluate the function in a single sparse grid point.

        """
        # create an empty array of size #Points
        self.f_eval = np.zeros(self.n_points)

        # loop over all grid points of the subspace
        for i in range(self.n_points):
            # evaluate the provided function in the current grid point
            self.f_eval[i] = function(self.points[i, :])

    def pass_contributions2higher_levels(self):
        """During sparse grid interpolation, this function passes contributions to all subspaces with higher level."""

        # loop over all subspaces of the SG (this can be made more efficient)
        for s in range(self.SG.nSubspaces):
            higherLevelSubspace = self.SG.subspace_list[s]

            # check if the higherLevelSubspace indeed has a higher level
            if np.sum(higherLevelSubspace.levels) > np.sum(self.levels):
                # loop over all points in the mother subspace and add contributions to lower levels
                for p in range(self.n_points):
                    higherLevelSubspace.lower_level_contributions += (
                        basis_nd(
                            higherLevelSubspace.points,
                            self.points[p, :],
                            self.levels,
                        )
                        * self.coeffs[p]
                    )


def evaluate_on_sparse_grid(
    args: typing.Tuple[
        np.ndarray,
        Model,
        np.ndarray,
        DataTransformation,
        np.ndarray,
        np.ndarray,
    ]
) -> np.ndarray:
    """This function is used to evaluate a function on a sparse grid in parallel. The input args is a tuple containing the function, the sparse grid and the number of processes to be used.

    Args:
        params (np.ndarray): The parameters to be evaluated.
        model(Model): The model used for the inference.
        data(np.ndarray): The data points used for the inference.
        data_transformation (DataTransformation): The data transformation used to normalize the data.
        data_stdevs(np.ndarray): The standard deviations of the data points. (Currently the kernel width, #TODO!)
        slice(np.ndarray): The slice defines for which dimensions of the grid points / paramater vectors the marginal density should be evaluated.

    Returns:
        np.ndarray: The density values for the given params.
    """

    params, model, data, data_transformation, data_stdevs, slice = args
    return evaluate_density(
        params, model, data, data_transformation, data_stdevs, slice
    )[1]


def inference_sparse_grid(
    model: Model,
    data: np.ndarray,
    data_transformation: DataTransformation,
    result_manager: ResultManager,
    slices: typing.List[np.ndarray],
    num_processes: int,
    num_levels: int = 5,
) -> typing.Tuple[
    typing.Dict[str, np.ndarray],
    typing.Dict[str, np.ndarray],
    typing.Dict[str, np.ndarray],
    ResultManager,
]:
    """Evaluates the transformed parameter density over a set of points resembling a sparse grid, thereby attempting parameter inference. If a data path is given, it is used to load the data for the model. Else, the default data path of the model is used.

    Args:
        model(Model): The model describing the mapping from parameters to data.
        data(np.ndarray): The data to be used for inference.
        data_transformation (DataTransformation): The data transformation used to normalize the data.
        num_processes(int): number of processes to use for parallel evaluation of the model.
        num_levels(int, optional): Maximum sparse grid level depth that mainly defines the number of points. Defaults to 5.

    Returns:
        Tuple[typing.Dict[str, np.ndarray], typing.Dict[str, np.ndarray], typing.Dict[str, np.ndarray], ResultManager]: The parameter samples, the corresponding simulation results, the corresponding density
        evaluations for each slice and the result manager used for the inference.

    """

    logger.warning(
        "The inference_sparse_grid function is not tested and not recommended for use."
    )
    data_stdevs = calc_kernel_width(data)

    # create the return dictionaries
    overall_params, overall_sim_results, overall_density_evals = {}, {}, {}

    for slice in slices:
        # build the sparse grid over [0,1]^param_dim
        grid = SparseGrid(slice.shape[0], num_levels)

        # get the model's parameter limits
        param_limits = model.param_limits

        # scale the sparse grid points from [0,1]^param_dim to the scaled parameter space
        scaledSparseGridPoints = param_limits[slice, 0] + grid.points * (
            param_limits[slice, 1] - param_limits[slice, 0]
        )

        # Create a pool of worker processes
        pool = get_context("spawn").Pool(processes=num_processes)
        tasks = zip(
            scaledSparseGridPoints,
            repeat(model),
            repeat(data),
            repeat(data_transformation),
            repeat(data_stdevs),
            repeat(slice),
        )

        # evaluate the probability density transformation for all sparse grid points in parallel
        results = pool.map(
            evaluate_on_sparse_grid,
            tasks,
        )

        # close the worker pool
        pool.close()
        pool.join()

        # convert the results to a numpy array
        # Take care! The results here are for single points, therefore we cant use np.concatenate
        results = np.vstack(results)

        data_dim = model.data_dim

        # save the results
        result_manager.save_overall(
            slice,
            results[:, 0 : slice.shape[0]],
            results[:, slice.shape[0] : slice.shape[0] + data_dim],
            results[:, slice.shape[0] + data_dim :],
        )
        slice_name = result_manager.get_slice_name(slice)
        overall_params[slice_name] = results[:, 0 : slice.shape[0]]
        overall_sim_results[slice_name] = results[
            :, slice.shape[0] : slice.shape[0] + data_dim
        ]
        overall_density_evals[slice_name] = results[
            :, slice.shape[0] + data_dim :
        ]
        result_manager.save_inference_information(
            slice=slice,
            model=model,
            inference_type=InferenceType.SPARSE_GRID,
            num_processes=num_processes,
            num_levels=num_levels,
        )

    return (
        overall_params,
        overall_sim_results,
        overall_density_evals,
        result_manager,
    )
