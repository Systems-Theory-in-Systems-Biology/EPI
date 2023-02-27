from epi import logger
from epi.core.model import Model
from epi.core.sampling import NUM_PROCESSES, NUM_RUNS, NUM_STEPS, NUM_WALKERS
import numpy as np
from enum import StrEnum, auto
import typing
import os

from epi.core.sampling import (
    concatenateEmceeSamplingResults,
    runEmceeSampling,
)

# Define an enum for the inference types: DenseGrid, MCMC
class InferenceType(StrEnum):
    DENSE_GRID = auto()
    MCMC = auto()


def inference(
    model: Model,
    data: typing.Union[str, os.PathLike, np.ndarray],
    inference_type: InferenceType,
    results_manager = None,
    slices: list[np.ndarray] = None,
    **kwargs,
):
    """Starts the parameter inference for the given model. If a data path is given, it is used to load the data for the model. Else, the default data path of the model is used.


    :param model: The model describing the mapping from parameters to data.
    :type model: Model
    :param dataPath: path to the data relative to the current working directory.
                      If None, the default path defined in the Model class initializer is used, defaults to None
    :type dataPath: str, optional
    :param numRuns: Number of independent runs, defaults to NUM_RUNS
    :type numRuns: int, optional
    :param numWalkers: Number of walkers for each run, influencing each other, defaults to NUM_WALKERS
    :type numWalkers: int, optional
    :param numSteps: Number of steps each walker does in each run, defaults to NUM_STEPS
    :type numSteps: int, optional
    :param numProcesses: number of processes to use, defaults to NUM_PROCESSES
    :type numProcesses: int, optional
    """

    if data is not None:
        model.setDataPath(data)
    else:
        logger.warning(
            f"No data path provided for this inference call. Using the data path of the model: {model.dataPath}"
        )

    # If no slice is given, compute full joint distribution, i.e. a slice with all parameters
    if slices is None:
        slice = np.arange(model.getCentralParam().shape[0])
        slices = [slice]

    if inference_type == InferenceType.DENSE_GRID:
        inference_dense_grid(model, data, results_manager, slices, **kwargs)
    elif inference_type == InferenceType.MCMC:
        inference_mcmc(model, data, results_manager, slices, **kwargs)
    else:
        raise NotImplementedError(
            f"The inference type {inference_type} is not implemented yet."
        )

def inference_dense_grid(model, data, results_manager = None, slices = None):
    raise NotImplementedError("Dense grid inference is not implemented yet.")

def inference_mcmc(model, data, results_manager = None, slices = None):
    # TODO: Use kwargs to pass the following parameters
    numRuns: int = NUM_RUNS,
    numWalkers: int = NUM_WALKERS,
    numSteps: int = NUM_STEPS,
    numProcesses: int = NUM_PROCESSES,

    for slice in slices:
        runEmceeSampling(
            model, slice, numRuns, numWalkers, numSteps, numProcesses
        )
    return concatenateEmceeSamplingResults(model)
    