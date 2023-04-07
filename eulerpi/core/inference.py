import os
import pathlib
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

from eulerpi.core.dense_grid import inference_dense_grid
from eulerpi.core.model import Model
from eulerpi.core.result_manager import ResultManager
from eulerpi.core.sampling import inference_mcmc
from eulerpi.core.sparsegrid import inference_sparse_grid


# Define an enum for the inference types: DenseGrid, MCMC
class InferenceType(Enum):
    """The type of inference to be used."""

    DENSE_GRID = 0  #: The dense grid inference uses a dense grid to evaluate the joint distribution.
    MCMC = 1  #: The MCMC inference uses a Markov Chain Monte Carlo sampler to sample from the joint distribution.
    SPARSE_GRID = 2  #: The sparse grid inference uses a sparse grid to evaluate the joint distribution. It is not tested and not recommended.


def inference(
    model: Model,
    data: Union[str, os.PathLike, np.ndarray],
    inference_type: InferenceType = InferenceType.MCMC,
    slices: Optional[list[np.ndarray]] = None,
    num_processes: int = 4,
    run_name: str = "default_run",
    result_manager: ResultManager = None,
    continue_sampling: bool = False,
    **kwargs,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    ResultManager,
]:
    """Starts the parameter inference for the given model. If a data path is given, it is used to load the data for the model. Else, the default data path of the model is used.

    Args:
        model(Model): The model describing the mapping from parameters to data.
        data(Union[str, os.PathLike, np.ndarray]): The data to be used for the inference. If a string is given, it is assumed to be a path to a file containing the data.
        inference_type(InferenceType, optional): The type of inference to be used. (Default value = InferenceType.MCMC)
        slices(list[np.ndarray], optional): A list of slices to be used for the inference. If None, the full joint distribution is computed. (Default value = None)
        num_processes(int, optional): The number of processes to be used for the inference. (Default value = 4)
        run_name(str): The name of the run. (Default value = "default_run")
        result_manager(ResultManager, optional): The result manager to be used for the inference. If None, a new result manager is created with default paths and saving methods. (Default value = None)
        continue_sampling(bool, optional): If True, the inference will continue sampling from the last saved point. (Default value = False)
        **kwargs: Additional keyword arguments to be passed to the inference function. The possible parameters depend on the inference type.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], ResultManager]: The parameter samples, the corresponding simulation results, the corresponding density
        evaluations for each slice and the result manager used for the inference.

    """

    # Load data from file if necessary
    if isinstance(data, (str, os.PathLike, pathlib.Path)):
        data = np.loadtxt(data, delimiter=",", ndmin=2)
    elif not isinstance(data, (np.ndarray, jnp.ndarray)):
        raise TypeError(
            f"The data argument must be a path to a file or a numpy array. The argument passed was of type {type(data)}."
        )

    slices = slices or [
        np.arange(model.param_dim)
    ]  # If no slice is given, compute full joint distribution, i.e. a slice with all parameters
    result_manager = result_manager or ResultManager(
        model.name, run_name
    )  # If no result_manager is given, create one with default paths

    if not continue_sampling:
        result_manager.delete_application_folder_structure(model, slices)
    result_manager.create_application_folder_structure(model, slices)

    if inference_type == InferenceType.DENSE_GRID:
        return inference_dense_grid(
            model=model,
            data=data,
            result_manager=result_manager,
            slices=slices,
            num_processes=num_processes,
            **kwargs,
        )
    elif inference_type == InferenceType.MCMC:
        return inference_mcmc(
            model=model,
            data=data,
            result_manager=result_manager,
            slices=slices,
            num_processes=num_processes,
            **kwargs,
        )
    elif inference_type == InferenceType.SPARSE_GRID:
        return inference_sparse_grid(
            model=model,
            data=data,
            result_manager=result_manager,
            slices=slices,
            num_processes=num_processes,
            **kwargs,
        )
    else:
        raise NotImplementedError(
            f"The inference type {inference_type} is not implemented yet."
        )
