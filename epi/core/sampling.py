"""Sampling methods for the EPI package.

This module provides functions to handle the sampling in EPI. It is based on the emcee package.

.. _emcee: https://emcee.readthedocs.io/en/stable/

Attributes:
    NUM_RUNS (int): Default number of runs of the emcee sampler.
    NUM_WALKERS (int): Default number of walkers in the emcee sampler.
    NUM_STEPS (int): Default number of steps each walker performs before storing the sub run.
    NUM_PROCESSES (int): Default number of parallel threads.

"""

import typing
from os import path

import emcee
import numpy as np
from schwimmbad import MultiPool

from epi import logger
from epi.core.kde import calc_kernel_width
from epi.core.model import Model
from epi.core.result_manager import ResultManager
from epi.core.transformations import eval_log_transformed_density

NUM_RUNS = 2
NUM_WALKERS = 10
NUM_STEPS = 2500
NUM_PROCESSES = 4

# TODO: This works on the blob
# Return the samples.
# return sampler.get_chain(discard=0, thin=1, flat=True)
# TODO: This stores the sample as 2d array in the format walker1_step1, walker2_step1, walker3_step1, walker1_step2, walker2_step2, walker3_step2, ...
# sampler_results = samplerBlob.reshape(
#     num_walkers * num_steps, sampling_dim + data_dim + 1
# )


def run_emcee_once(
    model: Model,
    data: np.ndarray,
    dataStdevs: np.ndarray,
    slice: np.ndarray,
    initialWalkerPositions: np.ndarray,
    num_walkers: int,
    num_steps: int,
    num_processes: int,
) -> np.ndarray:
    """Run the emcee particle swarm sampler once.

    Args:
        model (Model): The model which will be sampled
        data (np.ndarray): data
        dataStdevs (np.ndarray): kernel width for the data
        slice (np.ndarray): slice of the parameter space which will be sampled
        initialWalkerPositions (np.ndarray): initial parameter values for the walkers
        num_walkers (int): number of particles in the particle swarm sampler
        num_steps (int): number of samples each particle performs before storing the sub run
        num_processes (int): number of parallel threads

    Returns:
        np.ndarray: samples from the transformed parameter density

    """

    global work

    def work(params):
        s = eval_log_transformed_density(
            params, model, data, dataStdevs, slice
        )
        return s

    pool = MultiPool(processes=num_processes)

    # define a custom move policy
    movePolicy = [
        (emcee.moves.WalkMove(), 0.1),
        (emcee.moves.StretchMove(), 0.1),
        (
            emcee.moves.GaussianMove(0.00001, mode="sequential", factor=None),
            0.8,
        ),
    ]
    # movePolicy = [(emcee.moves.GaussianMove(0.00001, mode='sequential', factor=None), 1.0)]
    sampling_dim = slice.shape[0]

    # Call the sampler for all parallel workers (possibly use arg moves = movePolicy)
    sampler = emcee.EnsembleSampler(
        num_walkers,
        sampling_dim,
        # eval_log_transformed_density,
        work,
        pool=pool,
        moves=movePolicy,
        # args=[model, data, dataStdevs, slice],
    )
    # Extract the final walker position and close the pool of worker processes.
    finalWalkerPositionsitions, _, _, _ = sampler.run_mcmc(
        initialWalkerPositions, num_steps, tune=True, progress=True
    )
    if pool is not None:
        pool.close()
        pool.join()

    # TODO: Keep as 3d array?
    # Should have shape (num_steps, num_walkers, param_dim+data_dim+1)
    sampler_results = sampler.get_blobs()
    data_dim = data.shape[1]
    sampler_results = sampler_results.reshape(
        num_steps * num_walkers, sampling_dim + data_dim + 1
    )
    sampler_results = sampler_results.reshape(
        num_walkers * num_steps, sampling_dim + data_dim + 1
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

    return sampler_results, finalWalkerPositionsitions


def run_emcee_sampling(
    model: Model,
    data: np.ndarray,
    slice: np.ndarray,
    result_manager: ResultManager,
    num_runs: int = NUM_RUNS,
    num_walkers: int = NUM_WALKERS,
    num_steps: int = NUM_STEPS,
    num_processes: int = NUM_PROCESSES,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a representative sample from the transformed parameter density using the emcee particle swarm sampler.
       Inital values are not stored in the chain and each file contains <num_steps> blocks of size num_walkers.

    Args:
        model (Model): The model which will be sampled
        data (np.ndarray): data
        slice (np.ndarray): slice of the parameter space which will be sampled
        result_manager (ResultManager): ResultManager which will store the results
        num_runs (int, optional): number of stored sub runs. Defaults to NUM_RUNS.
        num_walkers (int, optional): number of particles in the particle swarm sampler. Defaults to NUM_WALKERS.
        num_steps (int, optional): number of samples each particle performs before storing the sub run. Defaults to NUM_STEPS.
        num_processes (int, optional): number of parallel threads. Defaults to NUM_PROCESSES.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: Array with all params, array with all data, array with all log probabilities

    """

    dataStdevs = calc_kernel_width(data)
    sampling_dim = slice.shape[0]
    central_param = model.central_param

    # Initialize each walker at a Gaussian-drawn random, slightly different parameter close to the central parameter.
    # TODO Make random variation of initial walker positions dependent on sampling limits?
    initialWalkerPositions = central_param[slice] + 0.002 * (
        np.random.rand(num_walkers, sampling_dim) - 0.5
    )

    # Count and print how many runs have already been performed for this model
    numExistingFiles = result_manager.count_emcee_sub_runs(slice)
    logger.debug(f"{numExistingFiles} existing files found")

    # Loop over the remaining sub runs and contiune the counter where it ended.
    for run in range(numExistingFiles, numExistingFiles + num_runs):
        logger.info(f"Run {run} of {num_runs}")

        # If there are current walker positions defined by runs before this one, use them.
        positionPath = result_manager.get_slice_path(slice) + "/currentPos.csv"
        if path.isfile(positionPath):
            initialWalkerPositions = np.loadtxt(
                positionPath,
                delimiter=",",
                ndmin=2,
            )
            logger.info(
                f"Continue sampling from saved sampler position in {positionPath}"
            )

        # Run the sampler.
        sampler_results, finalWalkerPositions = run_emcee_once(
            model,
            data,
            dataStdevs,
            slice,
            initialWalkerPositions,
            num_walkers,
            num_steps,
            num_processes,
        )

        result_manager.save_run(
            model, slice, run, sampler_results, finalWalkerPositions
        )

    (
        overallParams,
        overallSimResults,
        overallDensityEvals,
    ) = concatenate_emcee_sampling_results(model, result_manager, slice)
    result_manager.save_overall(
        slice, overallParams, overallSimResults, overallDensityEvals
    )
    return overallParams, overallSimResults, overallDensityEvals


def concatenate_emcee_sampling_results(
    model: Model, result_manager: ResultManager, slice: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate many sub runs of the emcee sampler to create 3 large files for sampled parameters, corresponding simulation results and density evaluations.
        These files are later used for result visualization.

    Args:
        model (Model): The model for which the results should be concatenated
        result_manager (ResultManager): ResultManager to load the results from
        slice (np.ndarray): slice for which the results should be concatenated

    Returns:
        typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: Array with all params, array with all data, array with all log probabilities

    """

    # Count and print how many sub runs are ready to be merged.
    numExistingFiles = result_manager.count_emcee_sub_runs(slice)
    logger.info(f"{numExistingFiles} existing files found for concatenation")

    densityFiles = (
        result_manager.get_slice_path(slice)
        + "/DensityEvals/densityEvals_{}.csv"
    )
    simResultsFiles = (
        result_manager.get_slice_path(slice) + "/SimResults/simResults_{}.csv"
    )
    paramFiles = result_manager.get_slice_path(slice) + "/Params/params_{}.csv"

    overallParams = np.vstack(
        [
            np.loadtxt(paramFiles.format(i), delimiter=",", ndmin=2)
            for i in range(numExistingFiles)
        ]
    )
    overallSimResults = np.vstack(
        [
            np.loadtxt(simResultsFiles.format(i), delimiter=",", ndmin=2)
            for i in range(numExistingFiles)
        ]
    )
    overallDensityEvals = np.hstack(
        [
            np.loadtxt(densityFiles.format(i), delimiter=",")
            for i in range(numExistingFiles)
        ]
    )

    return overallParams, overallSimResults, overallDensityEvals


def calc_walker_acceptance(
    model: Model,
    slice: np.ndarray,
    num_walkers: int,
    num_burn_samples: int,
    result_manager: ResultManager,
):
    """Calculate the acceptance ratio for each individual walker of the emcee chain.
    This is especially important to find "zombie" walkers, that are never moving.

    Args:
        model (Model): The model for which the acceptance ratio should be calculated
        slice (np.ndarray): slice for which the acceptance ratio should be calculated
        num_walkers (int): number of walkers in the emcee chain
        num_burn_samples (int): number of samples that were ignored at the beginning of each chain
        result_manager (ResultManager): ResultManager to load the results from

    Returns:
        np.ndarray: Array with the acceptance ratio for each walker

    """

    # load the emcee parameter chain
    params = np.loadtxt(
        result_manager.get_slice_path(slice) + "/OverallParams.csv",
        delimiter=",",
        ndmin=2,
    )[num_burn_samples:, :]

    # calculate the number of steps each walker walked
    # subtract 1 because we count the steps between the parameters
    num_steps = int(params.shape[0] / num_walkers) - 1

    # Unflatten the parameter chain and count the number of accepted steps for each walker
    params = params.reshape(num_steps + 1, num_walkers, model.param_dim)

    # Build a boolean array that is true if the parameters of the current step are the same as the parameters of the next step and sum over it
    # If the parameters are the same, the step is not accepted and we add 0 to the number of accepted steps
    # If the parameters are different, the step is accepted and we add 1 to the number of accepted steps
    numAcceptedSteps = np.sum(
        np.any(params[1:, :, :] != params[:-1, :, :], axis=2),
        axis=0,
    )

    # calculate the acceptance ratio by dividing the number of accepted steps by the overall number of steps
    acceptanceRatios = numAcceptedSteps / num_steps

    return acceptanceRatios
