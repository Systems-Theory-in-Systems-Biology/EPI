from itertools import repeat
from multiprocessing import get_context
from typing import Tuple

import numpy as np

from eulerpi.data_transformations import DataTransformation
from eulerpi.evaluation.density import DensityEvaluator, get_DensityEvaluator
from eulerpi.evaluation.kde import KDE
from eulerpi.grids.grid_factory import construct_grid
from eulerpi.models import BaseModel
from eulerpi.result_manager import ResultManager


def evaluate_function_on_grid_points_iterative(
    grid_points: np.ndarray,
    de: DensityEvaluator,
):
    n_points = grid_points.shape[0]
    result = np.zeros((n_points, de.dim_out))

    for i, gridPoint in enumerate(grid_points):
        _, param_data_density = de(gridPoint)
        result[i] = param_data_density
    return result


def evaluate_function_on_grid_points_batched(
    grid_points: np.ndarray,
    de: DensityEvaluator,
    num_processes: int,
    load_balancing_safety_faktor: int = 1,
):
    n_points = grid_points.shape[0]
    n_chunks = min(n_points, num_processes * load_balancing_safety_faktor)
    grid_chunks = np.array_split(grid_points, n_chunks)

    with get_context("spawn").Pool(processes=num_processes) as pool:
        # Use zip and repeat for lazy argument pairing
        results = pool.starmap(
            evaluate_function_on_grid_points_iterative,
            zip(grid_chunks, repeat(de)),
        )
    results = np.concatenate(results)

    return results


def grid_inference(
    model: BaseModel,
    data_transformation: DataTransformation,
    kde: KDE,
    slice: np.ndarray,
    result_manager: ResultManager,
    num_processes: int,
    grid_type: str = "EQUIDISTANT",
    load_balancing_safety_faktor: int = 1,
    # grid_detail: Union[int, np.ndarray] = 5,
    # num_levels: int = 5,
    num_grid_points=10,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function runs a dense grid evaluation for the given model and data.

    Args:
        model(BaseModel): The model for which the evaluation should be performed.
        data_transformation (DataTransformation): The data transformation used to normalize the data.
        slice(np.ndarray): The slice for which the evaluation should be performed.
        result_manager(ResultManager): The result manager that should be used to save the results.
        num_processes(int): The number of processes that should be used for the evaluation. (Default value = NUM_PROCESSES)
        grid_detail(Union[int, list[np.ndarray]]): The number of grid points for each dimension or another parameter defining the grid resolution
        grid_type(str): The type of grid that should be used. (Default value = "EQUIDISTANT")
        load_balancing_safety_faktor(int): Split the grid into num_processes * load_balancing_safety_faktor chunks.
            This will ensure that each process can be loaded with a similar amount of work if the run time difference between the evaluations
            does not exceed the load_balancing_safety_faktor. (Default value = 1)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The parameter samples, the corresponding simulation results, the corresponding density
        evaluations for the given slice.

    Raises:
        ValueError: If the dimension of the numbers of grid points does not match the number of parameters in the slice.
        ValueError: If the grid type is unknown.

    """

    limits = np.atleast_2d(model.param_limits)[slice, :]
    # TODO: This is only here, because some of the arguments get their default values just in the sampling_inference / grid_inference, not already in inference
    result_manager.save_inference_information(
        slice=slice,
        model=model,
        inference_type="GRID",  # TODO: Dont depend on details!
        num_processes=num_processes,
        grid_type=grid_type,
        load_balancing_safety_faktor=load_balancing_safety_faktor,
        num_grid_points=num_grid_points,
    )
    if grid_type == "SPARSE" and num_grid_points > 8:
        raise ValueError(
            "Sparse Grid with more than 8 levels is not supported for now"
        )

    grid = construct_grid(grid_type, limits, num_grid_points)
    density_evaluator = get_DensityEvaluator(
        model, data_transformation, kde, slice
    )

    grid_points = grid.grid_points
    if num_processes > 1:
        results = evaluate_function_on_grid_points_batched(
            grid_points,
            density_evaluator,
            num_processes,
            load_balancing_safety_faktor,
        )
    else:
        results = evaluate_function_on_grid_points_iterative(
            grid_points, density_evaluator
        )

    n_p = slice.shape[0]
    params = results[:, :n_p]
    sim_res = results[:, n_p : n_p + model.data_dim]
    densities = results[:, -1]

    result_manager.save_overall(
        slice,
        params,
        sim_res,
        densities,
    )
    return params, sim_res, densities
