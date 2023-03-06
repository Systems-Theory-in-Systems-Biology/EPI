import os
import pathlib
import typing
from enum import Enum

import jax.numpy as jnp
import numpy as np

from epi.core.dense_grid import NUM_GRID_POINTS, runDenseGridEvaluation
from epi.core.model import Model
from epi.core.result_manager import ResultManager
from epi.core.sampling import (
    NUM_PROCESSES,
    NUM_RUNS,
    NUM_STEPS,
    NUM_WALKERS,
    runEmceeSampling,
)


# Define an enum for the inference types: DenseGrid, MCMC
class InferenceType(Enum):
    DENSE_GRID = 0
    MCMC = 1


def inference(
    model: Model,
    data: typing.Union[str, os.PathLike, np.ndarray],
    inference_type: InferenceType = InferenceType.MCMC,
    slices: list[np.ndarray] = None,
    run_name: str = "default_run",
    result_manager=None,
    continue_sampling: bool = False,
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
    :param
    """

    # Load data from file if necessary
    assert data is not None
    if (
        isinstance(data, str)
        or isinstance(data, os.PathLike)
        or isinstance(data, pathlib.Path)
    ):
        data = np.loadtxt(data, delimiter=",", ndmin=2)
    elif isinstance(data, np.ndarray) or isinstance(data, jnp.ndarray):
        pass
    else:
        raise TypeError(
            f"The params argument has to be either a path to a file or a numpy array. The passed argument was of type {type(data)}"
        )
    # If no slice is given, compute full joint distribution, i.e. a slice with all parameters
    if slices is None:
        slice = np.arange(model.paramDim)
        slices = [slice]

    # If no result_manager is given, create one with default paths
    if result_manager is None:
        result_manager = ResultManager(model.name, run_name)

    if not continue_sampling:
        result_manager.deleteApplicationFolderStructure(model, slices)

    result_manager.createApplicationFolderStructure(model, slices)

    if inference_type == InferenceType.DENSE_GRID:
        inference_dense_grid(model, data, result_manager, slices, **kwargs)
    elif inference_type == InferenceType.MCMC:
        inference_mcmc(model, data, result_manager, slices, **kwargs)
    else:
        raise NotImplementedError(
            f"The inference type {inference_type} is not implemented yet."
        )


def inference_dense_grid(
    model,
    data,
    result_manager: ResultManager = None,
    slices: np.ndarray = None,
    allNumsGridPoints: typing.Union[int, list[np.ndarray]] = NUM_GRID_POINTS,
    numProcesses: int = NUM_PROCESSES,
):
    """This function runs a dense grid evaluation for the given model and data. # TODO document properly"""

    # If the number of grid points is given as an int, it is assumed to be the same for all parameters
    if isinstance(allNumsGridPoints, int):
        homogenousNumGridPoints = allNumsGridPoints
        numGridPoints = []
        for slice in slices:
            allNumsGridPoints.append(
                homogenousNumGridPoints * np.ones(slices[slice].shape[0])
            )
    elif isinstance(allNumsGridPoints, list[np.ndarray]):
        pass
    else:
        raise TypeError(
            f"The numGridPoints argument has to be either an int or a list of arrays. The passed argument was of type {type(numGridPoints)}"
        )
    for slice, numGridPoints in zip(slices, allNumsGridPoints):
        runDenseGridEvaluation(
            model,
            data,
            slice,
            result_manager,
            numGridPoints,
            numProcesses,
        )


def inference_mcmc(
    model,
    data,
    result_manager=None,
    slices=None,
    numRuns: int = NUM_RUNS,
    numWalkers: int = NUM_WALKERS,
    numSteps: int = NUM_STEPS,
    numProcesses: int = NUM_PROCESSES,
):

    for slice in slices:
        runEmceeSampling(
            model,
            data,
            slice,
            result_manager,
            numRuns,
            numWalkers,
            numSteps,
            numProcesses,
        )
