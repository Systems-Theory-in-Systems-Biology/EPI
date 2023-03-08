import os
import pathlib
import typing
from enum import Enum

import jax.numpy as jnp
import numpy as np

from epi import logger
from epi.core.dense_grid import NUM_GRID_POINTS, runDenseGridEvaluation
from epi.core.model import Model
from epi.core.result_manager import ResultManager
from epi.core.sampling import (
    NUM_PROCESSES,
    NUM_RUNS,
    NUM_STEPS,
    NUM_WALKERS,
    calcWalkerAcceptance,
    runEmceeSampling,
)


# Define an enum for the inference types: DenseGrid, MCMC
class InferenceType(Enum):
    """The type of inference to be used.

    Args:
        Enum: The enum class.

    Attributes:
        DENSE_GRID: The dense grid inference.
        MCMC: The MCMC inference.
    """

    # TODO: Sparse grid inference
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

    Args:
        model(Model): The model describing the mapping from parameters to data.
        data(typing.Union[str, os.PathLike, np.ndarray]): The data to be used for the inference. If a string is given, it is assumed to be a path to a file containing the data.
        inference_type(InferenceType): The type of inference to be used. (Default value = InferenceType.MCMC)
        slices(list[np.ndarray]): A list of slices to be used for the inference. If None, the full joint distribution is computed. (Default value = None)
        run_name(str): The name of the run. (Default value = "default_run")
        result_manager(ResultManager): The result manager to be used for the inference. If None, a new result manager is created with default paths and saving methods. (Default value = None)
        continue_sampling(bool): If True, the inference will continue sampling from the last saved point. (Default value = False)
        **kwargs: Additional keyword arguments to be passed to the inference function. The possible parameters depend on the inference type.

    Returns:

    """

    # Load data from file if necessary
    if isinstance(data, (str, os.PathLike, pathlib.Path)):
        data = np.loadtxt(data, delimiter=",", ndmin=2)
    elif not isinstance(data, (np.ndarray, jnp.ndarray)):
        raise TypeError(
            f"The data argument must be a path to a file or a numpy array. The argument passed was of type {type(data)}."
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
    model: Model,
    data: np.ndarray,
    result_manager: ResultManager,
    slices: np.ndarray,
    allNumsGridPoints: typing.Union[int, list[np.ndarray]] = NUM_GRID_POINTS,
    numProcesses: int = NUM_PROCESSES,
):
    """This function runs a dense grid evaluation for the given model and data. The grid points are distributed evenly over the parameter space.

    Args:
        model (Model): The model describing the mapping from parameters to data.
        data (np.ndarray): The data to be used for the inference.
        result_manager (ResultManager): The result manager to be used for the inference.
        slices (np.ndarray): A list of slices to be used for the inference.
        allNumsGridPoints (typing.Union[int, list[np.ndarray]], optional): The number of grid points to be used for each parameter. If an int is given, it is assumed to be the same for all parameters. Defaults to NUM_GRID_POINTS.
        numProcesses (int, optional): The number of processes to be used for the inference. Defaults to NUM_PROCESSES.

    Raises:
        TypeError: If the numGridPoints argument has the wrong type.
    """

    # If the number of grid points is given as an int, we construct a list of arrays with the same number of grid points for each parameter in the slice
    if isinstance(allNumsGridPoints, int):
        allNumsGridPoints = [
            np.full(len(slice), allNumsGridPoints) for slice in slices
        ]
    elif isinstance(allNumsGridPoints, list[np.ndarray]):
        pass
    else:
        raise TypeError(
            f"The numGridPoints argument has to be either an int or a list of arrays. The passed argument was of type {type(allNumsGridPoints)}"
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
    result_manager,
    slices,
    numRuns: int = NUM_RUNS,
    numWalkers: int = NUM_WALKERS,
    numSteps: int = NUM_STEPS,
    numProcesses: int = NUM_PROCESSES,
    calcWalkerAcceptanceB: bool = False,
):
    """This function runs a MCMC sampling for the given model and data.

    Args:
        model (Model): The model describing the mapping from parameters to data.
        data (np.ndarray): The data to be used for the inference.
        result_manager (ResultManager): The result manager to be used for the inference.
        slices (np.ndarray): A list of slices to be used for the inference.
        numRuns (int, optional): The number of runs to be used for the inference. Defaults to NUM_RUNS.
        numWalkers (int, optional): The number of walkers to be used for the inference. Defaults to NUM_WALKERS.
        numSteps (int, optional): The number of steps to be used for the inference. Defaults to NUM_STEPS.
        numProcesses (int, optional): The number of processes to be used for the inference. Defaults to NUM_PROCESSES.
        calcWalkerAcceptanceB (bool, optional): If True, the acceptance rate of the walkers is calculated and printed. Defaults to False.

    """

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
        if calcWalkerAcceptanceB:
            acceptance = calcWalkerAcceptance(
                model, result_manager, numWalkers, numBurnSamples=0
            )
            logger.info(f"Acceptance rate for slice {slice}: {acceptance}")
