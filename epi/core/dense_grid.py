from multiprocessing import Pool

import numpy as np

from epi.core.model import Model
from epi.core.result_manager import ResultManager
from epi.core.sampling import NUM_PROCESSES, calc_kernel_width
from epi.core.transformations import evaluate_density

NUM_GRID_POINTS = 10


def generate_grid(num_grid_points: np.ndarray, limits: np.ndarray, flat=False):
    """Generate a grid with the given number of grid points for each dimension.

    Args:
        num_grid_points(np.ndarray): The number of grid points for each dimension.
        limits(np.ndarray): The limits for each dimension.
        flat(bool): If True, the grid is returned as a flat array. If False, the grid is returned as a list of arrays, one for each dimension. (Default value = False)

    Returns:
        np.ndarray: The grid containing the grid points.

    """
    ndim = num_grid_points.size
    axes = [
        np.linspace(limits[i][0], limits[i][1], num=num_grid_points[i])
        for i in range(ndim)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    if flat:
        # TODO: Check if this is equivalent to the old code:  grid = grid.T.reshape(-1, slice.shape[0])
        # TODO: Can this code be simplified?
        return np.array(mesh).reshape(ndim, -1).T
    else:
        return mesh


def run_dense_grid_evaluation(
    model: Model,
    data: np.ndarray,
    slice: np.ndarray,
    result_manager: ResultManager,
    num_grid_points: np.ndarray,
    num_processes=NUM_PROCESSES,
) -> None:
    """This function runs a dense grid evaluation for the given model and data.

    Args:
        model(Model): The model for which the evaluation should be performed.
        data(np.ndarray): The data for which the evaluation should be performed.
        slice(np.ndarray): The slice for which the evaluation should be performed.
        result_manager(ResultManager): The result manager that should be used to save the results.
        num_grid_points(np.ndarray): The number of grid points for each dimension.
        num_processes(int): The number of processes that should be used for the evaluation. (Default value = NUM_PROCESSES)

    Raises:
        ValueError: If the dimension of the numbers of grid points does not match the number of parameters in the slice.

    """

    if slice.shape[0] != num_grid_points.shape[0]:
        raise ValueError(
            f"The dimension of the numbers of grid points {num_grid_points} does not match the number of parameters in the slice {slice}"
        )
    limits = model.param_limits
    dataStdevs = calc_kernel_width(data)

    grid = generate_grid(num_grid_points, limits, flat=True)

    # Split the grid into chunks that can be evaluated by each process
    grid_chunks = np.array_split(grid, num_processes)

    # Define a function which evaluates the density for a given grid chunk
    global evaluate_on_grid_chunk  # Needed to make this function pickleable

    def evaluate_on_grid_chunk(args):
        grid_chunk, model, data, dataStdevs, slice = args
        # Init the result array
        evaluation_results = np.zeros(
            (grid_chunk.shape[0], data.shape[1] + slice.shape[0] + 1)
        )
        # Evaluate the grid points
        for i, gridPoint in enumerate(grid_chunk):
            density, param_simRes_density = evaluate_density(
                gridPoint, model, data, dataStdevs, slice
            )
            evaluation_results[i] = param_simRes_density
        return evaluation_results

    pool = Pool(processes=num_processes)
    results = np.zeros((grid.shape[0], data.shape[1] + slice.shape[0] + 1))
    for i, result in enumerate(
        pool.imap(
            evaluate_on_grid_chunk,
            [
                (grid_chunks[i], model, data, dataStdevs, slice)
                for i in range(num_processes)
            ],
        )
    ):
        results[
            i * grid_chunks[i].shape[0] : (i + 1) * grid_chunks[i].shape[0]
        ] = result[1]
    pool.close()
    pool.join()

    result_manager.save_overall(
        slice, results[:, 0], results[:, 1], results[:, 2]
    )
