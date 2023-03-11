from enum import Enum
from multiprocessing import Pool
from typing import Union

import numpy as np
from numpy.polynomial.chebyshev import chebpts1

from epi.core.model import Model
from epi.core.result_manager import ResultManager
from epi.core.sampling import calc_kernel_width
from epi.core.transformations import evaluate_density


class DenseGridType(Enum):
    """The type of grid to be used."""

    EQUIDISTANT = 0  #: The equidistant grid has the same distance between two grid points in each dimension.
    CHEBYSHEV = 1  #: The Chebyshev grid is a tensor product of Chebyshev polynomial roots. They are optimal for polynomial interpolation and quadrature.


def generate_chebyshev_grid(
    num_grid_points: np.ndarray, limits: np.ndarray, flatten=False
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
    num_grid_points: np.ndarray, limits: np.ndarray, flatten=False
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


def run_dense_grid_evaluation(
    model: Model,
    data: np.ndarray,
    slice: np.ndarray,
    result_manager: ResultManager,
    num_grid_points: np.ndarray,
    dense_grid_type: DenseGridType,
    num_processes: int,
    load_balancing_safety_faktor: int,
) -> None:
    """This function runs a dense grid evaluation for the given model and data.

    Args:
        model(Model): The model for which the evaluation should be performed.
        data(np.ndarray): The data for which the evaluation should be performed.
        slice(np.ndarray): The slice for which the evaluation should be performed.
        result_manager(ResultManager): The result manager that should be used to save the results.
        num_grid_points(np.ndarray): The number of grid points for each dimension.
        dense_grid_type(DenseGridType): The type of grid that should be used. (Default value = DenseGridType.EQUIDISTANT)
        num_processes(int): The number of processes that should be used for the evaluation. (Default value = NUM_PROCESSES)
        load_balancing_safety_faktor(int): Split the grid into num_processes * load_balancing_safety_faktor chunks.
            This will ensure that each process can be loaded with a similar amount of work if the run time difference between the evaluations
            does not exceed the load_balancing_safety_faktor. (Default value = 4)

    Raises:
        ValueError: If the dimension of the numbers of grid points does not match the number of parameters in the slice.
        ValueError: If the grid type is unknown.

    """

    if slice.shape[0] != num_grid_points.shape[0]:
        raise ValueError(
            f"The dimension of the numbers of grid points {num_grid_points} does not match the number of parameters in the slice {slice}"
        )
    limits = model.param_limits
    data_stdevs = calc_kernel_width(data)

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
    # Calc cumsum for indexing
    grid_chunks_cumsum = np.cumsum(
        [0] + [grid_chunk.shape[0] for grid_chunk in grid_chunks]
    )
    # Define a function which evaluates the density for a given grid chunk
    global evaluate_on_grid_chunk  # Needed to make this function pickleable

    def evaluate_on_grid_chunk(args):
        grid_chunk, model, data, data_stdevs, slice = args
        # Init the result array
        evaluation_results = np.zeros(
            (grid_chunk.shape[0], data.shape[1] + slice.shape[0] + 1)
        )
        # Evaluate the grid points
        for i, gridPoint in enumerate(grid_chunk):
            density, param_sim_res_density = evaluate_density(
                gridPoint, model, data, data_stdevs, slice
            )
            evaluation_results[i] = param_sim_res_density
        return evaluation_results

    pool = Pool(processes=num_processes)
    results = np.zeros((grid.shape[0], data.shape[1] + slice.shape[0] + 1))
    for i, result in enumerate(
        pool.imap(
            evaluate_on_grid_chunk,
            [
                (grid_chunks[i], model, data, data_stdevs, slice)
                for i in range(len(grid_chunks))
            ],
        )
    ):
        results[grid_chunks_cumsum[i] : grid_chunks_cumsum[i + 1]] = result

    pool.close()
    pool.join()

    result_manager.save_overall(
        slice,
        results[:, 0 : data.shape[1]],
        results[:, data.shape[1] : data.shape[1] + slice.shape[0]],
        results[:, data.shape[1] + slice.shape[0] :],
    )


def inference_dense_grid(
    model: Model,
    data: np.ndarray,
    result_manager: ResultManager,
    slices: list[np.ndarray],
    num_processes: int,
    num_grid_points: Union[int, list[np.ndarray]] = 10,
    dense_grid_type: DenseGridType = DenseGridType.EQUIDISTANT,
    load_balancing_safety_faktor: int = 4,
) -> None:
    """This function runs a dense grid evaluation for the given model and data. The grid points are distributed evenly over the parameter space.

    Args:
        model (Model): The model describing the mapping from parameters to data.
        data (np.ndarray): The data to be used for the inference.
        result_manager (ResultManager): The result manager to be used for the inference.
        slices (np.ndarray): A list of slices to be used for the inference.
        num_processes (int): The number of processes to be used for the inference.
        num_grid_points (Union[int, list[np.ndarray]], optional): The number of grid points to be used for each parameter. If an int is given, it is assumed to be the same for all parameters. Defaults to 10.
        load_balancing_safety_faktor (int, optional): Split the grid into num_processes * load_balancing_safety_faktor chunks. Defaults to 4.

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
    for slice, n_points in zip(slices, num_grid_points):
        run_dense_grid_evaluation(
            model=model,
            data=data,
            slice=slice,
            result_manager=result_manager,
            num_grid_points=n_points,
            dense_grid_type=dense_grid_type,
            num_processes=num_processes,
            load_balancing_safety_faktor=load_balancing_safety_faktor,
        )
