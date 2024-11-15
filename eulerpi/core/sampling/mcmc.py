"""Module implementing the :py:func:`inference <eulerpi.core.inference.inference>` with a Markov Chain Monte Carlo(MCMC) sampling.

The inference with a sampling based approach approximates the the (joint/marginal) parameter distribution(s) by calculating a parameter Markov Chain
using multiple walkers in parallel. This module is currently based on the emcee package.

.. _emcee: https://emcee.readthedocs.io/en/stable/

.. note::

    The functions in this module are mainly intended for internal use and are accessed by :func:`inference <eulerpi.core.inference>` function.
    Read the documentation of :func:`inference_mcmc <inference_mcmc>` to learn more about the available options for the MCMC based inference.
"""

import typing
import warnings
from multiprocessing import get_context
from os import path

import emcee
import numpy as np

from eulerpi import logger
from eulerpi.core.data_transformations import DataTransformation
from eulerpi.core.evaluation.kde import KDE
from eulerpi.core.evaluation.transformations import (
    eval_log_transformed_density,
)
from eulerpi.core.models import BaseModel
from eulerpi.core.result_manager import ResultManager

INFERENCE_NAME = "MCMC"


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


def start_subrun(
    model: BaseModel,
    data_transformation: DataTransformation,
    kde: KDE,
    slice: np.ndarray,
    initial_walker_positions: np.ndarray,
    num_walkers: int,
    num_steps: int,
    num_processes: int,
) -> np.ndarray:
    """Run the emcee particle swarm sampler once.

    Args:
        model (BaseModel): The model which will be sampled
        data_transformation (DataTransformation): The data transformation used to normalize the data.
        kde(KDE): The density estimator which should be used to estimate the data density.
        slice (np.ndarray): slice of the parameter space which will be sampled
        initial_walker_positions (np.ndarray): initial parameter values for the walkers
        num_walkers (int): number of particles in the particle swarm sampler
        num_steps (int): number of samples each particle performs before storing the sub run
        num_processes (int): number of parallel threads

    Returns:
        np.ndarray: samples from the transformed parameter density

    """
    sampling_dim = slice.shape[0]

    # Create a pool of worker processes
    pool = get_context("spawn").Pool(processes=num_processes)

    # Call the sampler for all parallel workers (possibly use arg moves = movePolicy)
    try:
        sampler = emcee.EnsembleSampler(
            num_walkers,
            sampling_dim,
            eval_log_transformed_density,
            pool=pool,
            args=[model, data_transformation, kde, slice],
        )
        # Extract the final walker position and close the pool of worker processes.
        final_walker_positions, _, _, _ = sampler.run_mcmc(
            initial_walker_positions, num_steps, tune=True, progress=True
        )
    except ValueError as e:
        # If the message equals "Probability function returned NaN."
        if "Probability function returned NaN" in str(e):
            raise ValueError(
                "Probability function returned NaN. "
                "You possibly have to exclude data dimensions which do not depend on the paramaters. "
                "In addition your parameters should not be linearly dependent."
            )
        else:
            raise e

    if pool is not None:
        pool.close()
        pool.join()

    # TODO: Keep as 3d array?
    # Should have shape (num_steps, num_walkers, param_dim+data_dim+1)
    sampler_results = sampler.get_blobs()
    data_dim = model.data_dim
    sampler_results = sampler_results.reshape(
        num_steps * num_walkers, sampling_dim + data_dim + 1
    )

    logger.info(
        f"The acceptance fractions of the emcee sampler per walker are: {np.round(sampler.acceptance_fraction, 2)}"
    )
    try:
        corrTimes = sampler.get_autocorr_time()
        logger.info(f"autocorrelation time: {corrTimes[0]}")
    except emcee.autocorr.AutocorrError as e:
        logger.warning(
            "The autocorrelation time could not be calculate reliable"
        )

    return sampler_results, final_walker_positions


def inference_mcmc(
    model: BaseModel,
    data_transformation: DataTransformation,
    kde: KDE,
    slice: np.ndarray,
    result_manager: ResultManager,
    num_processes: int,
    num_runs: int = 1,
    num_walkers: int = 10,
    num_steps: int = 2500,
    num_burn_in_samples: typing.Optional[int] = None,
    thinning_factor: typing.Optional[int] = None,
    get_walker_acceptance: bool = False,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a representative sample from the transformed parameter density using the emcee particle swarm sampler.
       Inital values are not stored in the chain and each file contains <num_steps> blocks of size num_walkers.

    Args:
        model (BaseModel): The model which will be sampled
        data_transformation (DataTransformation): The data transformation used to normalize the data.
        kde(KDE): The density estimator which should be used to estimate the data density.
        slice (np.ndarray): slice of the parameter space which will be sampled
        result_manager (ResultManager): ResultManager which will store the results
        num_processes (int): number of parallel threads.
        num_runs (int): number of stored sub runs.
        num_walkers (int): number of particles in the particle swarm sampler.
        num_steps (int): number of samples each particle performs before storing the sub run.
        num_burn_in_samples(int): Number of samples that will be deleted (burned) per chain (i.e. walker). Only for mcmc inference.
        thinning_factor (int): thinning factor for the samples.
        get_walker_acceptance (bool): If True, the acceptance rate of the walkers is calculated and printed. Defaults to False.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: Array with all params, array with all data, array with all log probabilities
        TODO check: are those really log probabilities?

    """
    if num_burn_in_samples is None:
        num_burn_in_samples = int(num_runs * num_steps * 0.1)
    if thinning_factor is None:
        thinning_factor = 1

    sampling_dim = slice.shape[0]

    # Initialize each walker at a Gaussian-drawn random, slightly different parameter close to the central parameter.
    # compute element-wise min of the difference between the central parameter and the lower sampling limit and the difference between the central parameter and the upper sampling limit
    d_min = np.minimum(
        model.central_param - model.param_limits[:, 0],
        model.param_limits[:, 1] - model.central_param,
    )
    initial_walker_positions = model.central_param[slice] + d_min[slice] * (
        np.random.rand(num_walkers, sampling_dim) - 0.5
    )

    # Count and print how many runs have already been performed for this model
    num_existing_files = result_manager.count_emcee_sub_runs(slice)
    logger.debug(f"{num_existing_files} existing files found")

    # Loop over the remaining sub runs and continue the counter where it ended.
    for i_subrun in range(num_existing_files, num_existing_files + num_runs):
        logger.info(f"Subrun {i_subrun} of {num_runs}")

        # If there are current walker positions defined by runs before this one, use them.
        position_path = (
            result_manager.get_slice_path(slice) + "/currentPos.csv"
        )
        if path.isfile(position_path):
            initial_walker_positions = np.loadtxt(
                position_path,
                delimiter=",",
                ndmin=2,
            )
            logger.info(
                f"Continue sampling from saved sampler position in {position_path}"
            )

        # Run the sampler.
        with warnings.catch_warnings():
            # This warning is raised when the model returned a -inf value for the log probability, e.g. because the parameters are out of bounds.
            # We want to ignore this warning, because the sampler will handle this case correctly.
            # NaN values and other errors are not affected by this.
            warnings.filterwarnings(
                "ignore",
                module="red_blue",
                category=RuntimeWarning,
                message="invalid value encountered in scalar subtract",
            )
            sampler_results, final_walker_positions = start_subrun(
                model,
                data_transformation,
                kde,
                slice,
                initial_walker_positions,
                num_walkers,
                num_steps,
                num_processes,
            )

        result_manager.save_subrun(
            model, slice, i_subrun, sampler_results, final_walker_positions
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
        num_burn_in_steps = int(num_steps * num_runs * 0.01)
        acceptance = calc_walker_acceptance(
            model, slice, num_walkers, num_burn_in_steps, result_manager
        )
        logger.info(f"Acceptance rate for slice {slice}: {acceptance}")

    return (
        overall_params,
        overall_sim_results,
        overall_density_evals,
    )
