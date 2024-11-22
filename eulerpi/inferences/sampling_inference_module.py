import os
from functools import partial
from typing import Optional, Tuple

import numpy as np

from eulerpi.data_transformations.data_transformation import DataTransformation
from eulerpi.evaluation.kde import KDE
from eulerpi.evaluation.transformation import evaluate_log_density
from eulerpi.function_wrappers import FunctionWithDimensions
from eulerpi.logger import logger
from eulerpi.models.base_model import BaseModel
from eulerpi.result_manager import ResultManager
from eulerpi.samplers.emcee_sampler import EmceeSampler
from eulerpi.samplers.sampler import Sampler


def calc_walker_acceptance(
    model: BaseModel,
    slice: np.ndarray,
    num_walkers: int,
    num_burn_in_samples: int,
    result_manager: ResultManager,
):
    """Calculate the acceptance ratio for each individual walker of the emcee chain.
    This is especially important to find "zombie" walkers, that are never moving.

    Args:
        model (BaseModel): The model for which the acceptance ratio should be calculated
        slice (np.ndarray): slice for which the acceptance ratio should be calculated
        num_walkers (int): number of walkers in the emcee chain
        num_burn_in_samples(int): Number of samples that will be deleted (burned) per chain (i.e. walker). Only for mcmc inference.
        result_manager (ResultManager): ResultManager to load the results from

    Returns:
        np.ndarray: Array with the acceptance ratio for each walker

    """
    thinning_factor = 1
    params, _, _ = result_manager.load_slice_inference_results(
        slice, num_burn_in_samples, thinning_factor
    )
    num_steps = (
        params.shape[0] // num_walkers - 1
    )  # calculate the number of steps per walker
    params = params.reshape(
        num_steps + 1, num_walkers, model.param_dim
    )  # Unflatten the parameter chain and count the number of accepted steps for each walker
    different_params = np.any(
        params[1:] != params[:-1], axis=2
    )  # Check for parameter changes between consecutive steps
    num_accepted_steps = np.sum(
        different_params, axis=0
    )  # Sum over changes to count accepted steps
    acceptance_ratio = num_accepted_steps / num_steps
    return acceptance_ratio


def combined_evaluation(param, model, data_transformation, kde, slice):
    param, sim_res, logdensity = evaluate_log_density(
        param, model, data_transformation, kde, slice
    )
    combined_result = np.concatenate([param, sim_res, np.array([logdensity])])
    return logdensity, combined_result


def get_LogDensityEvaluator(
    model: BaseModel,
    data_transformation: DataTransformation,
    kde: KDE,
    slice: np.ndarray,
):
    logdensity_blob_function = partial(
        combined_evaluation,
        model=model,
        data_transformation=data_transformation,
        kde=kde,
        slice=slice,
    )

    param_dim = slice.shape[0]
    output_dim = (1, param_dim + model.data_dim + 1)
    return FunctionWithDimensions(
        logdensity_blob_function, param_dim, output_dim
    )


def _load_sampler_position(position_path: os.PathLike):
    # If there are current walker positions defined by runs before this one, use them.

    if os.path.isfile(position_path):
        initial_walker_positions = np.loadtxt(
            position_path,
            delimiter=",",
            ndmin=2,
        )
        return initial_walker_positions
    return None


def sampling_inference(
    model: BaseModel,
    data_transformation: DataTransformation,
    kde: KDE,
    slice: np.ndarray,
    result_manager: ResultManager,
    num_processes: int,
    sampler: Sampler = None,
    num_walkers: int = 10,
    num_steps: int = 2500,
    num_burn_in_samples: Optional[int] = None,
    thinning_factor: Optional[int] = None,
    get_walker_acceptance: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a representative sample from the transformed parameter density using the emcee particle swarm sampler.
       Inital values are not stored in the chain and each file contains <num_steps> blocks of size num_walkers.

    Args:
        model (BaseModel): The model which will be sampled
        data_transformation (DataTransformation): The data transformation used to normalize the data.
        kde(KDE): The density estimator which should be used to estimate the data density.
        slice (np.ndarray): slice of the parameter space which will be sampled
        result_manager (ResultManager): ResultManager which will store the results
        num_processes (int): number of parallel threads.
        num_walkers (int): number of particles in the particle swarm sampler.
        num_steps (int): number of samples each particle performs before storing the sub run.
        num_burn_in_samples(int): Number of samples that will be deleted (burned) per chain (i.e. walker). Only for mcmc inference.
        thinning_factor (int): thinning factor for the samples.
        get_walker_acceptance (bool): If True, the acceptance rate of the walkers is calculated and printed. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Array with all params, array with all data, array with all log probabilities

    """
    result_manager.append_inference_information(
        slice=slice,
        num_walkers=num_walkers,
        num_steps=num_steps,
        num_burn_in_samples=num_burn_in_samples,
        thinning_factor=thinning_factor,
        get_walker_acceptance=get_walker_acceptance,
    )

    logdensity_blob_function = get_LogDensityEvaluator(
        model, data_transformation, kde, slice
    )

    if num_burn_in_samples is None:
        num_burn_in_samples = int(num_steps * 0.1)
    if thinning_factor is None:
        thinning_factor = 1

    sampling_dim = slice.shape[0]

    if sampler is None:
        sampler = EmceeSampler(sampling_dim=sampling_dim, logger=logger)

    position_path = result_manager.get_slice_path(slice) + "/currentPos.csv"
    if last_position := _load_sampler_position(position_path) is not None:
        initial_walker_positions = last_position
        logger.info(
            f"Continue sampling from saved sampler position in {position_path}"
        )
    else:
        # Initialize each walker at a Gaussian-drawn random, slightly different parameter close to the central parameter.
        # compute element-wise min of the difference between the central parameter and the lower sampling limit and the difference between the central parameter and the upper sampling limit
        d_min = np.minimum(
            model.central_param - model.param_limits[:, 0],
            model.param_limits[:, 1] - model.central_param,
        )
        initial_walker_positions = model.central_param[slice] + d_min[
            slice
        ] * (np.random.rand(num_walkers, sampling_dim) - 0.5)

    # Count and print how many runs have already been performed for this model
    num_existing_files = result_manager.count_sub_runs(slice)
    logger.debug(f"{num_existing_files} existing files found")

    # Run the sampler.
    logger.info(f"Starting sampler run {num_existing_files}")
    sampler_results, final_walker_positions = sampler.run(
        logdensity_blob_function,
        initial_walker_positions,
        num_walkers,
        num_steps,
        num_processes,
    )

    result_manager.save_subrun(
        model,
        slice,
        num_existing_files,
        sampler_results,
        final_walker_positions,
    )

    (
        overall_params,
        overall_sim_results,
        overall_density_evals,
    ) = result_manager.load_slice_inference_results(
        slice, num_burn_in_samples, thinning_factor
    )
    result_manager.save_overall(
        slice,
        overall_params,
        overall_sim_results,
        overall_density_evals,
    )

    if get_walker_acceptance:
        acceptance = calc_walker_acceptance(
            model, slice, num_walkers, num_burn_in_samples, result_manager
        )
        logger.info(f"Acceptance rate for slice {slice}: {acceptance}")

    return (
        overall_params,
        overall_sim_results,
        overall_density_evals,
    )
