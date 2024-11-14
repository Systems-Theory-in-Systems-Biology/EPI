"""Module implementing the :py:func:`inference <eulerpi.core.inference.inference>` with a dense grid.

The inference with a dense grid approximates the (joint/marginal) parameter distribution(s) on a dense grid.

.. note::

    The functions in this module are mainly intended for internal use and are accessed by :func:`inference <eulerpi.core.inference>` function.
    Read the documentation of :func:`inference_dense_grid <eulerpi.core.dense_grid.inference_dense_grid>` to learn more
    about the available options for the inference with a dense grid.
"""

import typing
from itertools import repeat
from multiprocessing import get_context
from typing import Dict, Tuple, Union

import numpy as np
from numpy.polynomial.chebyshev import chebpts1

from eulerpi.core.data_transformations import DataTransformation
from eulerpi.core.dense_grid_types import DenseGridType
from eulerpi.core.inference_types import InferenceType
from eulerpi.core.model import Model
from eulerpi.core.result_manager import ResultManager
from eulerpi.core.sampling import calc_kernel_width
from eulerpi.core.transformations import evaluate_density


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


def evaluate_on_grid_chunk(
    args: typing.Tuple[
        np.ndarray,
        Model,
        np.ndarray,
        DataTransformation,
        np.ndarray,
        np.ndarray,
    ]
) -> np.ndarray:
    """Define a function which evaluates the density for a given grid chunk. The input args contains the grid chunk, the model, the data, the data_stdevs and the slice.

    Args:
        grid_chunk(np.ndarray): The grid chunk contains the grid points (parameter vectors) for which the density should be evaluated.
        model(Model): The model used for the inference.
        data(np.ndarray): The data points used for the inference.
        data_transformation (DataTransformation): The data transformation used to normalize the data.
        data_stdevs(np.ndarray): The standard deviations of the data points. (Currently the kernel width, #TODO!)
        slice(np.ndarray): The slice defines for which dimensions of the grid points / paramater vectors the marginal density should be evaluated.

    Returns:
        np.ndarray: The evaluation results for the given grid chunk. It is a vector, containing the parameters in the first columns, the simulation results in the second columns and the density evaluations in the last columns.
    """
    grid_chunk, model, data, data_transformation, data_stdevs, slice = args

    # Init the result array
    evaluation_results = np.zeros(
        (grid_chunk.shape[0], model.data_dim + slice.shape[0] + 1)
    )
    # Evaluate the grid points
    for i, gridPoint in enumerate(grid_chunk):
        density, param_sim_res_density = evaluate_density(
            gridPoint, model, data, data_transformation, data_stdevs, slice
        )
        evaluation_results[i] = param_sim_res_density
    return evaluation_results


def run_dense_grid_evaluation(
    model: Model,
    data: np.ndarray,
    data_transformation: DataTransformation,
    slice: np.ndarray,
    result_manager: ResultManager,
    num_grid_points: np.ndarray,
    dense_grid_type: DenseGridType,
    num_processes: int,
    load_balancing_safety_faktor: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function runs a dense grid evaluation for the given model and data.

    Args:
        model(Model): The model for which the evaluation should be performed.
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

    data_stdevs = calc_kernel_width(data)
    # Split the grid into chunks that can be evaluated by each process
    grid_chunks = np.array_split(
        grid, num_processes * load_balancing_safety_faktor
    )

    pool = get_context("spawn").Pool(processes=num_processes)
    tasks = zip(
        grid_chunks,
        repeat(model),
        repeat(data),
        repeat(data_transformation),
        repeat(data_stdevs),
        repeat(slice),
    )
    results = pool.map(evaluate_on_grid_chunk, tasks)
    pool.close()
    pool.join()
    results = np.concatenate(results)

    data_dim = model.data_dim

    result_manager.save_overall(
        slice,
        results[:, 0 : slice.shape[0]],
        results[:, slice.shape[0] : slice.shape[0] + data_dim],
        results[:, slice.shape[0] + data_dim :],
    )
    return (
        results[:, 0 : slice.shape[0]],
        results[:, slice.shape[0] : slice.shape[0] + data_dim],
        results[:, slice.shape[0] + data_dim :],
    )


def inference_dense_grid(
    model: Model,
    data: np.ndarray,
    data_transformation: DataTransformation,
    result_manager: ResultManager,
    slices: list[np.ndarray],
    num_processes: int,
    num_grid_points: Union[int, list[np.ndarray]] = 10,
    dense_grid_type: DenseGridType = DenseGridType.EQUIDISTANT,
    load_balancing_safety_faktor: int = 4,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    ResultManager,
]:
    """This function runs a dense grid inference for the given model and data.

    Args:
        model (Model): The model describing the mapping from parameters to data.
        data (np.ndarray): The data to be used for the inference.
        data_transformation (DataTransformation): The data transformation used to normalize the data.
        result_manager (ResultManager): The result manager to be used for the inference.
        slices (np.ndarray): A list of slices to be used for the inference.
        num_processes (int): The number of processes to be used for the inference.
        num_grid_points (Union[int, list[np.ndarray]], optional): The number of grid points to be used for each parameter. If an int is given, it is assumed to be the same for all parameters. Defaults to 10.
        dense_grid_type (DenseGridType, optional): The type of grid that should be used. Defaults to DenseGridType.EQUIDISTANT.
        load_balancing_safety_faktor (int, optional): Split the grid into num_processes * load_balancing_safety_faktor chunks. Defaults to 4.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], ResultManager]: The parameter samples, the corresponding simulation results, the corresponding density
        evaluations for each slice and the result manager used for the inference.

    Raises:
        TypeError: If the num_grid_points argument has the wrong type.
    """

    # If the number of grid points is given as an int, we construct a list of arrays with the same number of grid points for each parameter in the slice
    if isinstance(num_grid_points, int):
        num_grid_points = [
            np.full(len(slice), num_grid_points) for slice in slices
        ]
    elif isinstance(num_grid_points, list[np.ndarray]):
        pass
    else:
        raise TypeError(
            f"The num_grid_points argument has to be either an int or a list of arrays. The passed argument was of type {type(num_grid_points)}"
        )
    # create the return dictionaries
    overall_params, overall_sim_results, overall_density_evals = {}, {}, {}

    for slice, n_points in zip(slices, num_grid_points):
        slice_name = result_manager.get_slice_name(slice)
        (
            overall_params[slice_name],
            overall_sim_results[slice_name],
            overall_density_evals[slice_name],
        ) = run_dense_grid_evaluation(
            model=model,
            data=data,
            data_transformation=data_transformation,
            slice=slice,
            result_manager=result_manager,
            num_grid_points=n_points,
            dense_grid_type=dense_grid_type,
            num_processes=num_processes,
            load_balancing_safety_faktor=load_balancing_safety_faktor,
        )
        result_manager.save_inference_information(
            slice=slice,
            model=model,
            inference_type=InferenceType.DENSE_GRID,
            num_processes=num_processes,
            load_balancing_safety_faktor=load_balancing_safety_faktor,
            num_grid_points=n_points,
            dense_grid_type=dense_grid_type,
        )
    return (
        overall_params,
        overall_sim_results,
        overall_density_evals,
        result_manager,
    )
