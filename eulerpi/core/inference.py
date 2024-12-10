"""The inference module provides the main interface to the eulerpi library in the form of the :py:func:`inference <inference>` function."""

import os
import pathlib
from typing import Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import psutil

from eulerpi.core.data_transformations import (
    DataNormalization,
    DataTransformation,
)
from eulerpi.core.dense_grid import inference_dense_grid
from eulerpi.core.inference_types import InferenceType
from eulerpi.core.models import BaseModel
from eulerpi.core.result_managers import OutputWriter, ResultReader
from eulerpi.core.result_managers import get_slice_name
from eulerpi.core.sampling import inference_mcmc
from eulerpi.core.sparsegrid import inference_sparse_grid


def inference(
    model: BaseModel,
    data: Union[str, os.PathLike, np.ndarray],
    inference_type: InferenceType = InferenceType.MCMC,
    slice: Optional[np.ndarray] = None,
    num_processes: Optional[int] = None,
    run_name: Optional[str] = None,
    output_writer: OutputWriter = None,
    continue_sampling: bool = False,
    data_transformation: DataTransformation = None,
    **kwargs,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    ResultReader,
]:
    """Starts the parameter inference for the given model and data.

    Args:
        model(BaseModel): The model describing the mapping from parameters to data.
        data(Union[str, os.PathLike, np.ndarray]): The data to be used for the inference. If a string is given, it is assumed to be a path to a file containing the data.
        inference_type(InferenceType, optional): The type of inference to be used. (Default value = InferenceType.MCMC)
        slice(np.ndarray, optional): The slice to be used for the inference. If None, the full joint distribution is computed. (Default value = None)
        num_processes(int, optional): The number of processes to be used for the inference. Per default the number of cores is used. (Default value = Non)
        run_name(str, optional): The name of the run. (Default value = None gives "default_run_[slice_name]")
        output_writer(OutputWriter, optional): The output writer to be used for the inference. If None, a new output writer is created with default paths and saving methods. (Default value = None)
        continue_sampling(bool, optional): If True, the inference will continue sampling from the last saved point. (Default value = False)
        data_transformation(DataTransformation): The data transformation to use. If None is passed, a DataNormalization will be applied. Pass DataIdentity to avoid the transformation of the data. (Default value = None)
        **kwargs: Additional keyword arguments to be passed to the inference function. The possible parameters depend on the inference type.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], ResultReader]: The parameter samples, the pushforward of the parameters, the corresponding density
        evaluations for each slice, and a result reader to (re)-read and manipulate the results.

    Examples:

    Inference with eulerpi only requires the model and the data.
    The following example shows how to run inference for the Covid example model.

    .. code-block:: python

        import numpy as np
        from eulerpi.examples.corona import Corona
        from eulerpi.core.inference import inference

        # generate 1000 artificial, 4D data points for the Covid example model
        data_scales = np.array([1.0, 5.0, 35.0, 2.0])
        data = (np.random.rand(1000, 4)+1.0)*data_scales

        # run inference only specifying the model and the data
        (param_sample, sim_res_sample, desities_dict, _) = inference(Corona(), data)

        # retrieve (for instance) the parameter samples by evaluating the parameter sample dictionary in the slice Q0Q1Q2 corresponding to all three joint parameter dimensions
        param_sample = param_sample_dict["Slice_Q0Q1Q2"]

    Of course, you can also specify additional parameters and import data from a file Data/Coronoa/data.csv.
    In addition, the result reader is a convenient and versatile interface to access the inference results.

    .. code-block:: python

        import numpy as np
        from eulerpi.examples.corona import Corona
        from eulerpi.core.inference import inference

        # run inference with additional arguments
        (_, _, _, res_reader) = inference(Corona(),
                                        data = "pathto/data.csv", # load data from a csv file location
                                        num_processes = 8, # use 8 processes in parallelization
                                        run_name = "second_run", # specify the run
                                        num_walkers = 50, # use 50 walkers during sampling
                                        num_steps = 200, # each walker performs 200 steps
                                        num_burn_in_samples = 20, # discard the first 20 steps of each walker
                                        thinning_factor = 10) # only use every 10th sample to avoid autocorrelation

        # use the result reader to retreive the inference results
        param_sample, sim_res_sample, density_evals = res_reader.load_inference_results()

    Principle Compoonent Analysis (PCA) can be useful to reduce the dimension of the data space and is supported by eulerpi.
    Grid based parameter density estimation is especially useful whenever parameter dimension can be assumed to be indepedent.
    In the following example, we assume parameter dimension 2 to be independent from dimensions 0 and 1.
    Therefore, we compute the joint density of dimensions 0 and 1 and the marginal density of dimension 2.
    Please note, that specifying 30 grid points for a 2D density estimation results in 900 grid points in total.
    The result reader is especially useful to access already computed inference results and only requires the model and run name.

    .. code-block:: python

        import numpy as np
        from eulerpi.examples.corona import Corona
        from eulerpi.core.inference import inference, InferenceType
        from eulerpi.core.data_transformation import DataPCA
        from eulerpi.core.result_managers import ResultReader

        model = Corona()
        data_transformation = DataPCA(data, n_components=model.param_dim)  # perform PCA on the data before inference

        inference(model,
                data = "pathto/data.csv",
                slice = [np.array([0,1]), np.array([2])], # specify joint and marginal parameter subdistributions we are interested in
                inference_type = InferenceType.DENSE_GRID, # use dense grid inference
                run_name = "grid_run",
                data_transformation = data_transformation,
                num_grid_points = 30) # use 30 grid points per parameter dimension

        # initiate the result reader using the model and run name to retreive the inference results computed above and stored offline
        grid_res_reader = ResultReader(model_name = Corona().name, run_name = "grid_run")

        # load the inference results for the joint distribution of parameter dimensions 0 and 1
        param_grid_dim01, pushforward_evals_grid_dim01, density_evals_dim01 = grid_res_reader.load_inference_results()

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

    if slice is None:
        slice = np.arange(
            model.param_dim
        )  # If no slice is given, compute full joint distribution, i.e. a slice with all parameters

    if not run_name:
        run_name = f"default_run_{get_slice_name(slice)}"

    output_writer = output_writer or OutputWriter(
        model.name, run_name
    )  # If no output writer is given, create one with default paths

    if not continue_sampling:
        output_writer.delete_output_folder_structure()
    output_writer.create_output_folder_structure()

    if not num_processes:
        num_processes = psutil.cpu_count(logical=False)

    if inference_type == InferenceType.DENSE_GRID:
        return inference_dense_grid(
            model=model,
            data=data,
            data_transformation=data_transformation,
            output_writer=output_writer,
            slice=slice,
            num_processes=num_processes,
            **kwargs,
        )
    elif inference_type == InferenceType.MCMC:
        return inference_mcmc(
            model=model,
            data=data,
            data_transformation=data_transformation,
            output_writer=output_writer,
            slice=slice,
            num_processes=num_processes,
            **kwargs,
        )
    elif inference_type == InferenceType.SPARSE_GRID:
        return inference_sparse_grid(
            model=model,
            data=data,
            data_transformation=data_transformation,
            output_writer=output_writer,
            slice=slice,
            num_processes=num_processes,
            **kwargs,
        )
    else:
        raise NotImplementedError(
            f"The inference type {inference_type} is not implemented yet."
        )
