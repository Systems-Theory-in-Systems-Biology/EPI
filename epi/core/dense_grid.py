from multiprocessing import Pool

import numpy as np

from epi.core.model import Model
from epi.core.result_manager import ResultManager
from epi.core.sampling import NUM_PROCESSES, calcKernelWidth
from epi.core.transformations import evaluateDensity

NUM_GRID_POINTS = 10


def generate_grid(numGridPoints: np.ndarray, limits: np.ndarray, flat=False):
    """Generate a grid with the given number of grid points for each dimension.

    Args:
        numGridPoints(np.ndarray): The number of grid points for each dimension.
        limits(np.ndarray): The limits for each dimension.
        flat(bool): If True, the grid is returned as a flat array. If False, the grid is returned as a list of arrays, one for each dimension. (Default value = False)

    Returns:
        np.ndarray: The grid containing the grid points.

    """
    ndim = numGridPoints.size
    axes = [
        np.linspace(limits[i][0], limits[i][1], num=numGridPoints[i])
        for i in range(ndim)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    if flat:
        # TODO: Check if this is equivalent to the old code:  grid = grid.T.reshape(-1, slice.shape[0])
        # TODO: Can this code be simplified?
        return np.array(mesh).reshape(ndim, -1).T
    else:
        return mesh


def runDenseGridEvaluation(
    model: Model,
    data: np.ndarray,
    slice: np.ndarray,
    result_manager: ResultManager,
    numGridPoints: np.ndarray,
    numProcesses=NUM_PROCESSES,
) -> None:
    """This function runs a dense grid evaluation for the given model and data.

    Args:
        model(Model): The model for which the evaluation should be performed.
        data(np.ndarray): The data for which the evaluation should be performed.
        slice(np.ndarray): The slice for which the evaluation should be performed.
        result_manager(ResultManager): The result manager that should be used to save the results.
        numGridPoints(np.ndarray): The number of grid points for each dimension.
        numProcesses(int): The number of processes that should be used for the evaluation. (Default value = NUM_PROCESSES)

    Raises:
        ValueError: If the dimension of the numbers of grid points does not match the number of parameters in the slice.

    """

    if slice.shape[0] != numGridPoints.shape[0]:
        raise ValueError(
            f"The dimension of the numbers of grid points {numGridPoints} does not match the number of parameters in the slice {slice}"
        )
    limits = model.paramLimits
    dataStdevs = calcKernelWidth(data)

    grid = generate_grid(numGridPoints, limits, flat=True)

    # TODO: Do not spawn a process for each grid point, but spawn a process for each core and let it evaluate multiple grid points
    pool = Pool(processes=numProcesses)
    results = []
    for result in pool.imap_unordered(
        evaluateDensity,
        [(point, model, data, dataStdevs, slice) for point in grid],
    ):
        trafoDensityEvaluation, evaluationResults = result
        results.append(result)
        # TODO save intermediate results in csv

    pool.close()
    pool.join()

    for result in results:
        trafoDensityEvaluation, evaluationResults = result
        # TODO save overall results in csv
