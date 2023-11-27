import os
import pathlib
from typing import Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import psutil

from eulerpi.core.data_transformation import (
    DataIdentity,
    DataNormalizer,
    DataPCA,
    DataTransformation,
)
from eulerpi.core.data_transformation_types import DataTransformationType
from eulerpi.core.dense_grid import inference_dense_grid
from eulerpi.core.inference_types import InferenceType
from eulerpi.core.model import Model
from eulerpi.core.model_check import basic_model_check
from eulerpi.core.result_manager import ResultManager
from eulerpi.core.sampling import inference_mcmc
from eulerpi.core.sparsegrid import inference_sparse_grid


def inference(
    model: Model,
    data: Union[str, os.PathLike, np.ndarray],
    inference_type: InferenceType = InferenceType.MCMC,
    slices: Optional[list[np.ndarray]] = None,
    num_processes: Optional[int] = None,
    run_name: str = "default_run",
    result_manager: ResultManager = None,
    continue_sampling: bool = False,
    data_transformation: DataTransformationType = DataTransformationType.Normalize,
    custom_data_transformation: Optional[DataTransformation] = None,
    n_components_pca: Optional[int] = None,
    check_model: bool = True,
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
        num_processes(int, optional): The number of processes to be used for the inference. Per default the number of cores is used. (Default value = Non)
        run_name(str): The name of the run. (Default value = "default_run")
        result_manager(ResultManager, optional): The result manager to be used for the inference. If None, a new result manager is created with default paths and saving methods. (Default value = None)
        continue_sampling(bool, optional): If True, the inference will continue sampling from the last saved point. (Default value = False)
        data_transformation(DataTransformationType): The type of data transformation to use. (Default value = DataTransformationType.Normalize)
        custom_data_transformation(DataTransformation, optional): The data transformation to be used for the inference. If None, a normalization is applied to the data. (Default value = None)
        n_components_pca(int, optional): If using the PCA as data_transformation, selects how many dimensions are kept in the pca. Per default the number of dimensions equals the dimension of the parameter space. (Default value = None)
        check_model(bool, optional): If True, the model is checked for basic functionality before attempting inference. (Default value = True)
        **kwargs: Additional keyword arguments to be passed to the inference function. The possible parameters depend on the inference type.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], ResultManager]: The parameter samples, the corresponding simulation results, the corresponding density
        evaluations for each slice and the result manager used for the inference.

    """

    # check model for basic functionality
    if check_model:
        basic_model_check(model)

    # Load data from file if necessary
    if isinstance(data, (str, os.PathLike, pathlib.Path)):
        data = np.loadtxt(data, delimiter=",", ndmin=2)
    elif not isinstance(data, (np.ndarray, jnp.ndarray)):
        raise TypeError(
            f"The data argument must be a path to a file or a numpy array. The argument passed was of type {type(data)}."
        )

    # Transform the data
    if data_transformation == DataTransformationType.Identity:
        data_transformation = DataIdentity()
    elif data_transformation == DataTransformationType.Normalize:
        data_transformation = DataNormalizer.from_data(data)
    elif data_transformation == DataTransformationType.PCA:
        n_components = n_components_pca or model.param_dim
        data_transformation = DataPCA.from_data(
            data=data, n_components=n_components
        )
    elif data_transformation == DataTransformationType.Custom:
        data_transformation = custom_data_transformation
        if not issubclass(custom_data_transformation, DataTransformation):
            raise TypeError(
                f"The custom_data_transformation must be an instance of a subclass from DataTransformation. It is of type {type(data_transformation)}."
            )
    else:
        raise TypeError(
            "The data_transformation must be one of the enum values of DataTransformationType."
        )
    data = data_transformation.transform(data)

    # TODO rename std_dev to kernel_width, adapt calculation of kernel width

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

    if inference_type == InferenceType.DENSE_GRID:
        return inference_dense_grid(
            model=model,
            data=data,
            data_transformation=data_transformation,
            result_manager=result_manager,
            slices=slices,
            num_processes=num_processes,
            **kwargs,
        )
    elif inference_type == InferenceType.MCMC:
        return inference_mcmc(
            model=model,
            data=data,
            data_transformation=data_transformation,
            result_manager=result_manager,
            slices=slices,
            num_processes=num_processes,
            **kwargs,
        )
    elif inference_type == InferenceType.SPARSE_GRID:
        return inference_sparse_grid(
            model=model,
            data=data,
            data_transformation=data_transformation,
            result_manager=result_manager,
            slices=slices,
            num_processes=num_processes,
            **kwargs,
        )
    else:
        raise NotImplementedError(
            f"The inference type {inference_type} is not implemented yet."
        )
