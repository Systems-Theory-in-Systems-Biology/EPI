"""Module implementing the :py:func:`inference <eulerpi.core.inference.inference>` with a dense grid.

The inference with a dense grid approximates the (joint/marginal) parameter distribution(s) on a dense grid.

.. note::

    The functions in this module are mainly intended for internal use and are accessed by :func:`inference <eulerpi.core.inference>` function.
    Read the documentation of :func:`inference_dense_grid <eulerpi.core.sampling.dense_grid.inference_dense_grid>` to learn more
    about the available options for the inference with a dense grid.
"""

import typing
from enum import Enum
from itertools import repeat
from multiprocessing import get_context
from typing import Tuple, Union

import numpy as np

from eulerpi.core.data_transformations import DataTransformation
from eulerpi.core.evaluation.kde import KDE
from eulerpi.core.evaluation.transformations import evaluate_density
from eulerpi.core.models import BaseModel
from eulerpi.core.result_manager import ResultManager

from .grid_generators import generate_chebyshev_grid, generate_regular_grid

INFERENCE_NAME = "DENSE_GRID"


class DenseGridType(Enum):
    """The available grid types for the :py:mod:`dense grid<eulerpi.core.sampling.dense_grid>` inference."""

    EQUIDISTANT = 0  #: The equidistant grid has the same distance between two grid points in each dimension.
    CHEBYSHEV = 1  #: The Chebyshev grid is a tensor product of Chebyshev polynomial roots. They are optimal for polynomial interpolation and quadrature.


def evaluate_on_grid_chunk(
    args: typing.Tuple[
        np.ndarray,
        BaseModel,
        DataTransformation,
        KDE,
        np.ndarray,
    ]
) -> np.ndarray:
    """Define a function which evaluates the density for a given grid chunk. The input args contains the grid chunk, the model, the data, the data_stdevs and the slice.

    Args:
        grid_chunk(np.ndarray): The grid chunk contains the grid points (parameter vectors) for which the density should be evaluated.
        model(BaseModel): The model used for the inference.
        data_transformation (DataTransformation): The data transformation used to normalize the data.
        kde (KDE): The kernel density estimator (or some other estimator implementing the __call__ function) to estimate the density at a data point
        slice(np.ndarray): The slice defines for which dimensions of the grid points / paramater vectors the marginal density should be evaluated.

    Returns:
        np.ndarray: The evaluation results for the given grid chunk. It is a vector, containing the parameters in the first columns, the simulation results in the second columns and the density evaluations in the last columns.
    """
    grid_chunk, model, data_transformation, kde, slice = args

    # Init the result array
    evaluation_results = np.zeros(
        (grid_chunk.shape[0], model.data_dim + slice.shape[0] + 1)
    )
    # Evaluate the grid points
    for i, gridPoint in enumerate(grid_chunk):
        density, param_sim_res_density = evaluate_density(
            gridPoint, model, data_transformation, kde, slice
        )
        evaluation_results[i] = param_sim_res_density
    return evaluation_results


def inference_dense_grid(
    model: BaseModel,
    data_transformation: DataTransformation,
    kde: KDE,
    slice: np.ndarray,
    num_processes: int,
    result_manager: ResultManager,
    num_grid_points: Union[int, list[np.ndarray]] = 10,
    dense_grid_type: DenseGridType = DenseGridType.EQUIDISTANT,
    load_balancing_safety_faktor: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function runs a dense grid evaluation for the given model and data.

    Args:
        model(BaseModel): The model for which the evaluation should be performed.
        data(np.ndarray): The data for which the evaluation should be performed.
        data_transformation (DataTransformation): The data transformation used to normalize the data.
        slice(np.ndarray): The slice for which the evaluation should be performed.
        result_manager(ResultManager): The result manager that should be used to save the results.
        num_grid_points(np.ndarray): The number of grid points for each dimension.
        dense_grid_type(DenseGridType): The type of grid that should be used. (Default value = DenseGridType.EQUIDISTANT)
        num_processes(int): The number of processes that should be used for the evaluation. (Default value = NUM_PROCESSES)
        load_balancing_safety_faktor(int): Split the grid into num_processes * load_balancing_safety_faktor chunks.
            This will ensure that each process can be loaded with a similar amount of work if the run time difference between the evaluations
            does not exceed the load_balancing_safety_faktor. (Default value = 4)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The parameter samples, the corresponding simulation results, the corresponding density
        evaluations for the given slice.

    Raises:
        ValueError: If the dimension of the numbers of grid points does not match the number of parameters in the slice.
        ValueError: If the grid type is unknown.

    """

    if slice.shape[0] != num_grid_points.shape[0]:
        raise ValueError(
            f"The dimension of the numbers of grid points {num_grid_points} does not match the number of parameters in the slice {slice}"
        )
    if model.param_limits.ndim == 1:
        limits = model.param_limits
    else:
        limits = model.param_limits[slice, :]

    if dense_grid_type == DenseGridType.CHEBYSHEV:
        grid = generate_chebyshev_grid(num_grid_points, limits, flatten=True)
    elif dense_grid_type == DenseGridType.EQUIDISTANT:
        grid = generate_regular_grid(num_grid_points, limits, flatten=True)
    else:
        raise ValueError(f"Unknown grid type: {dense_grid_type}")

    # Split the grid into chunks that can be evaluated by each process
    grid_chunks = np.array_split(
        grid, num_processes * load_balancing_safety_faktor
    )
    tasks = zip(
        grid_chunks,
        repeat(model),
        repeat(data_transformation),
        repeat(kde),
        repeat(slice),
    )
    with get_context("spawn").Pool(processes=num_processes) as pool:
        results = pool.map(evaluate_on_grid_chunk, tasks)
    results = np.concatenate(results)

    data_dim = model.data_dim

    n_p = slice.shape[0]
    params = results[:, :n_p]
    sim_res = results[:, n_p : n_p + data_dim]
    densities = results[:, -1]

    result_manager.save_overall(
        slice,
        params,
        sim_res,
        densities,
    )
    return params, sim_res, densities
