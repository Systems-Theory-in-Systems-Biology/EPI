"""The inference module provides the main interface to the eulerpi library in the form of the :py:func:`inference <inference>` function."""

import os
import pathlib
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import psutil

from eulerpi.data_transformations import DataNormalization, DataTransformation
from eulerpi.evaluation import KDE, GaussKDE
from eulerpi.inferences.grid_inference import inference as grid_inference
from eulerpi.inferences.sampling_inference import (
    inference as sampling_inference,
)
from eulerpi.models import BaseModel
from eulerpi.result_manager import ResultManager
from eulerpi.samplers.mcmc import start_subrun as start_mcmc_subrun


class InferenceType(Enum):
    """Available modes for the :py:func:`inference <eulerpi.inference.inference>` function."""

    GRID = 0  #: The grid inference uses a grid to evaluate the joint distribution.
    SAMPLING = 1  #: The SAMPLING / MCMC inference uses a Markov Chain Monte Carlo sampler to sample from the joint distribution.


def inference(
    model: BaseModel,
    data: Union[str, os.PathLike, np.ndarray],
    inference_type: InferenceType = InferenceType.SAMPLING,
    slices: Optional[list[np.ndarray]] = None,
    num_processes: Optional[int] = None,
    run_name: str = "default_run",
    result_manager: ResultManager = None,
    continue_sampling: bool = False,
    data_transformation: DataTransformation = None,
    kde: KDE = None,
    **kwargs,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    ResultManager,
]:
    """Starts the parameter inference for the given model and data.

    Args:
        model(BaseModel): The model describing the mapping from parameters to data.
        data(Union[str, os.PathLike, np.ndarray]): The data to be used for the inference. If a string is given, it is assumed to be a path to a file containing the data.
        inference_type(InferenceType, optional): The type of inference to be used. (Default value = InferenceType.SAMPLING)
        slices(list[np.ndarray], optional): A list of slices to be used for the inference. If None, the full joint distribution is computed. (Default value = None)
        num_processes(int, optional): The number of processes to be used for the inference. Per default the number of cores is used. (Default value = None)
        run_name(str): The name of the run. (Default value = "default_run")
        result_manager(ResultManager, optional): The result manager to be used for the inference. If None, a new result manager is created with default paths and saving methods. (Default value = None)
        continue_sampling(bool, optional): If True, the inference will continue sampling from the last saved point. (Default value = False)
        data_transformation(DataTransformation): The data transformation to use. If None is passed, a DataNormalization will be applied. Pass DataIdentity to avoid the transformation of the data. (Default value = None)
        kde(KDE): The density estimator which should be used to estimate the data density. If None is passed, a GaussianKDE will be used. (Default value = None)
        **kwargs: Additional keyword arguments to be passed to the inference function. The possible parameters depend on the inference type.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], ResultManager]: The parameter samples, the corresponding simulation results, the corresponding density
        evaluations for each slice and the result manager used for the inference.

    Examples:

    Inference with eulerpi only requires the model and the data.
    The following example shows how to run inference for the Covid example model.

    .. code-block:: python

        import numpy as np
        from eulerpi.examples.corona import Corona
        from eulerpi import inference

        # generate 1000 artificial, 4D data points for the Covid example model
        data_scales = np.array([1.0, 5.0, 35.0, 2.0])
        data = (np.random.rand(1000, 4)+1.0)*data_scales

        # run inference only specifying the model and the data
        (param_sample_dict, sim_res_sample_dict, desities_dict, _) = inference(Corona(), data)

        # retrieve (for instance) the parameter samples by evaluating the parameter sample dictionary in the slice Q0Q1Q2 corresponding to all three joint parameter dimensions
        param_sample = param_sample_dict["Slice_Q0Q1Q2"]

    Of course, you can also specify additional parameters and import data from a file Data/Coronoa/data.csv.
    In addition, the result manager is a convenient and versatile interface to access the inference results.

    .. code-block:: python

        import numpy as np
        from eulerpi.examples.corona import Corona
        from eulerpi import inference

        # run inference with additional arguments
        (_, _, _, res_manager) = inference(Corona(),
                                        data = "pathto/data.csv", # load data from a csv file location
                                        num_processes = 8, # use 8 processes in parallelization
                                        run_name = "second_run", # specify the run
                                        num_walkers = 50, # use 50 walkers during sampling
                                        num_steps = 200, # each walker performs 200 steps
                                        num_burn_in_samples = 20, # discard the first 20 steps of each walker
                                        thinning_factor = 10) # only use every 10th sample to avoid autocorrelation

        # use the result manager to retreive the inference results
        param_sample, sim_res_sample, density_evals = res_manager.load_slice_inference_results(slice = np.array([0,1,2]))

    Principle Compoonent Analysis (PCA) can be useful to reduce the dimension of the data space and is supported by eulerpi.
    Grid based parameter density estimation is especially useful whenever parameter dimension can be assumed to be indepedent.
    In the following example, we assume parameter dimension 2 to be independent from dimensions 0 and 1.
    Therefore, we compute the joint density of dimensions 0 and 1 and the marginal density of dimension 2.
    Please note, that specifying 30 grid points for a 2D density estimation results in 900 grid points in total.
    The result manager is especially useful to access already computed inference results and only requires the model and run name.

    .. code-block:: python

        import numpy as np
        from eulerpi.examples.corona import Corona
        from eulerpi import inference, InferenceType
        from eulerpi.data_transformation import DataPCA
        from eulerpi.result_manager import ResultManager

        model = Corona()
        data_transformation = DataPCA(data, n_components=model.param_dim)  # perform PCA on the data before inference

        inference(model,
                data = "pathto/data.csv",
                slices = [np.array([0,1]), np.array([2])], # specify joint and marginal parameter subdistributions we are interested in
                inference_type = InferenceType.GRID, # use grid inference
                run_name = "grid_run",
                data_transformation = data_transformation,
                num_grid_points = 30) # use 30 grid points per parameter dimension

        # initiate the result manager using the model and run name to retreive the inference results computed above and stored offline
        grid_res_manager = ResultManager(model_name = Corona().name, run_name = "grid_run")

        # load the inference results for the joint distribution of parameter dimensions 0 and 1
        param_grid_dim01, sim_res_grid_dim01, density_evals_dim01 = grid_res_manager.load_slice_inference_results(slice = np.array([0,1]))

    """

    # Load data from file if necessary
    if isinstance(data, (str, os.PathLike, pathlib.Path)):
        data = np.loadtxt(data, delimiter=",", ndmin=2)
    elif not isinstance(data, (np.ndarray, jnp.ndarray)):
        raise TypeError(
            f"The data argument must be a path to a file or a numpy array. The argument passed was of type {type(data)}."
        )

    # Assign the DataNormalization as default value
    if data_transformation is None:
        data_transformation = DataNormalization(data)

    if not isinstance(data_transformation, DataTransformation):
        raise TypeError(
            f"The data_transformation must be an instance of a subclass of DataTransformation. It is of type {type(data_transformation)}."
        )
    data = data_transformation.transform(data)

    if kde is None:
        kde = GaussKDE(data)
    if not isinstance(kde, KDE):
        raise TypeError(
            f"The kde must be an instance of a subclass of KDE. It is of type {type(kde)}."
        )

    slices = slices or [
        np.arange(model.param_dim)
    ]  # If no slice is given, compute full joint distribution, i.e. a slice with all parameters
    result_manager = result_manager or ResultManager(
        model.name, run_name, slices
    )  # If no result_manager is given, create one with default paths

    if not continue_sampling:
        result_manager.delete_application_folder_structure()
    result_manager.create_application_folder_structure()

    if not num_processes:
        num_processes = psutil.cpu_count(logical=False)

    # create the return dictionaries
    overall_params, overall_sim_results, overall_density_evals = {}, {}, {}

    for slice in slices:
        slice_name = result_manager.get_slice_name(slice)
        if inference_type == InferenceType.GRID:
            params, sim_results, densities = grid_inference(
                model=model,
                data_transformation=data_transformation,
                kde=kde,
                slice=slice,
                result_manager=result_manager,
                num_processes=num_processes,
                **kwargs,
            )
        elif inference_type == InferenceType.SAMPLING:
            params, sim_results, densities = sampling_inference(
                model,
                data_transformation,
                kde,
                slice,
                result_manager,
                num_processes,
                start_mcmc_subrun,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                f"The inference type {inference_type} is not implemented yet."
            )

        overall_params[slice_name] = params
        overall_sim_results[slice_name] = sim_results
        overall_density_evals[slice_name] = densities

    return (
        overall_params,
        overall_sim_results,
        overall_density_evals,
        result_manager,
    )
