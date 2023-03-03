import typing
from multiprocessing import Pool

import numpy as np

from epi.core.model import Model
from epi.core.result_manager import ResultManager
from epi.core.sampling import NUM_PROCESSES

NUM_GRID_POINTS = 10


def evaluateInPoint(
    model: Model, data: np.ndarray, slice: np.ndarray, point: np.ndarray
) -> typing.Tuple[float, np.ndarray]:
    """This function evaluates the model in the given point and returns the log likelihood and the simulated data."""
    # TODO document properly
    # TODO implement
    raise (NotImplementedError)


def runDenseGridEvaluation(
    model: Model,
    data: np.ndarray,
    slice: np.ndarray,
    result_manager: ResultManager,
    numGridPoints: np.ndarray,
    numProcesses=NUM_PROCESSES,
) -> None:
    """This function runs a dense grid evaluation for the given model and data. # TODO document properly"""

    if slice.shape[0] != numGridPoints.shape[0]:
        raise ValueError(
            f"The dimension of the numbers of grid points {numGridPoints} does not match the number of parameters in the slice {slice}"
        )
    limits = model.paramLimits
    grid = np.array(
        np.meshgrid(
            *[
                np.linspace(limits[i, 0], limits[i, 1], numGridPoints[i])
                for i in range(slice.shape[0])
            ]
        )
    ).T.reshape(-1, slice.shape[0])
    pool = Pool(processes=numProcesses)
    raise (NotImplementedError)
