from itertools import repeat
from multiprocessing import get_context

import numpy as np

from eulerpi.function_wrappers import FunctionWithDimensions


def evaluate_function_on_grid_points_iterative(
    grid_points: np.ndarray,
    func: FunctionWithDimensions,
):
    n_points = grid_points.shape[0]
    result = np.zeros((n_points, func.dim_out))

    for i, gridPoint in enumerate(grid_points):
        result[i] = func(gridPoint)
    return result


def evaluate_function_on_grid_points_multiprocessing(
    grid_points: np.ndarray,
    func: FunctionWithDimensions,
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
            zip(grid_chunks, repeat(func)),
        )
    results = np.concatenate(results)

    return results
