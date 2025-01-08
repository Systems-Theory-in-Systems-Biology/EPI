from functools import partial
from typing import Tuple

import numpy as np

from eulerpi.data_transformations.data_transformation import DataTransformation
from eulerpi.evaluation.kde import KDE
from eulerpi.evaluation.transformation import evaluate_density
from eulerpi.function_wrappers import FunctionWithDimensions
from eulerpi.grids.equidistant_grid import EquidistantGrid
from eulerpi.grids.grid import Grid
from eulerpi.grids.grid_evaluation import (
    evaluate_function_on_grid_points_iterative,
    evaluate_function_on_grid_points_multiprocessing,
)
from eulerpi.models.base_model import BaseModel
from eulerpi.result_managers import OutputWriter, ResultReader, PathManager


def combined_evaluation(param, model, data_transformation, kde, slice):
    param, pushforward_evals, density = evaluate_density(
        param, model, data_transformation, kde, slice
    )
    combined_result = np.concatenate(
        [param, pushforward_evals, np.array([density])]
    )
    return combined_result


def get_DensityEvaluator(
    model: BaseModel,
    data_transformation: DataTransformation,
    kde: KDE,
    slice: np.ndarray,
):
    combined_evaluation_function = partial(
        combined_evaluation,
        model=model,
        data_transformation=data_transformation,
        kde=kde,
        slice=slice,
    )

    param_dim = slice.shape[0]
    output_dim = param_dim + model.data_dim + 1
    return FunctionWithDimensions(
        combined_evaluation_function, param_dim, output_dim
    )


def grid_inference(
    model: BaseModel,
    data_transformation: DataTransformation,
    kde: KDE,
    slice: np.ndarray,
    output_writer: OutputWriter,
    num_processes: int,
    grid: Grid = None,
    num_grid_points=10,
    load_balancing_safety_factor: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function runs a dense grid evaluation for the given model and data.

    Args:
        model(BaseModel): The model for which the evaluation should be performed.
        data_transformation (DataTransformation): The data transformation used to normalize the data.
        slice(np.ndarray): The slice for which the evaluation should be performed.
        output_writer(OutputWriter): The output writer that should be used to save the results.
        num_processes(int): The number of processes that should be used for the evaluation. (Default value = NUM_PROCESSES)
        grid_detail(Union[int, list[np.ndarray]]): The number of grid points for each dimension or another parameter defining the grid resolution
        grid_type(str): The type of grid that should be used. (Default value = "EQUIDISTANT")
        load_balancing_safety_factor(int): Split the grid into num_processes * load_balancing_safety_factor chunks.
            This will ensure that each process can be loaded with a similar amount of work if the run time difference between the evaluations
            does not exceed the load_balancing_safety_factor. (Default value = 1)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The parameter samples, the corresponding pushforward evaluations, the corresponding density
        evaluations for the given slice.

    Raises:
        ValueError: If the dimension of the numbers of grid points does not match the number of parameters in the slice.
        ValueError: If the grid type is unknown.

    """
    output_writer.append_inference_information(
        slice=slice,
        grid=type(grid).__name__,
        load_balancing_safety_factor=load_balancing_safety_factor,
        num_grid_points=num_grid_points,
    )

    if grid is None:
        limits = np.atleast_2d(model.param_limits)[slice, :]
        grid = EquidistantGrid(limits, num_grid_points)
    density_evaluator = get_DensityEvaluator(
        model, data_transformation, kde, slice
    )

    grid_points = grid.grid_points
    if num_processes > 1:
        results = evaluate_function_on_grid_points_multiprocessing(
            grid_points,
            density_evaluator,
            num_processes,
            load_balancing_safety_factor,
        )
    else:
        results = evaluate_function_on_grid_points_iterative(
            grid_points, density_evaluator
        )

    n_p = slice.shape[0]
    params = results[:, :n_p]
    pushforward_evals = results[:, n_p : n_p + model.data_dim]
    density_evals = results[:, -1]

    output_writer.save_grid_based_run(
        params=params,
        pushforward_evals=pushforward_evals,
        density_evals=density_evals,
    )
    return params, pushforward_evals, density_evals
