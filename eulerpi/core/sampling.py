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

from eulerpi import logger
from eulerpi.core.kde import calc_kernel_width
from eulerpi.core.model import Model
from eulerpi.core.result_manager import ResultManager
from eulerpi.core.transformations import eval_log_transformed_density

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
    data_stdevs: np.ndarray,
    slice: np.ndarray,
    initial_walker_positions: np.ndarray,
    num_walkers: int,
    num_steps: int,
    num_processes: int,
) -> np.ndarray:
    """Run the emcee particle swarm sampler once.

    Args:
        model (Model): The model which will be sampled
        data (np.ndarray): data
        data_stdevs (np.ndarray): kernel width for the data
        slice (np.ndarray): slice of the parameter space which will be sampled
        initial_walker_positions (np.ndarray): initial parameter values for the walkers
        num_walkers (int): number of particles in the particle swarm sampler
        num_steps (int): number of samples each particle performs before storing the sub run
        num_processes (int): number of parallel threads

    Returns:
        np.ndarray: samples from the transformed parameter density

    """

    global work

    def work(params):
        s = eval_log_transformed_density(
            params, model, data, data_stdevs, slice
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
    try:
        sampler = emcee.EnsembleSampler(
            num_walkers,
            sampling_dim,
            # eval_log_transformed_density,
            work,
            pool=pool,
            moves=movePolicy,
            # args=[model, data, data_stdevs, slice],
        )
        # Extract the final walker position and close the pool of worker processes.
        final_walker_positions, _, _, _ = sampler.run_mcmc(
            initial_walker_positions, num_steps, tune=True, progress=True
        )
    except ValueError as e:
        # If the message equals "Probability function returned NaN."
        if str(e) == "Probability function returned NaN.":
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

    return sampler_results, final_walker_positions


def run_emcee_sampling(
    model: Model,
    data: np.ndarray,
    slice: np.ndarray,
    result_manager: ResultManager,
    num_processes: int,
    num_runs: int,
    num_walkers: int,
    num_steps: int,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a representative sample from the transformed parameter density using the emcee particle swarm sampler.
       Inital values are not stored in the chain and each file contains <num_steps> blocks of size num_walkers.

    Args:
        model (Model): The model which will be sampled
        data (np.ndarray): data
        slice (np.ndarray): slice of the parameter space which will be sampled
        result_manager (ResultManager): ResultManager which will store the results
        num_processes (int): number of parallel threads.
        num_runs (int): number of stored sub runs.
        num_walkers (int): number of particles in the particle swarm sampler.
        num_steps (int): number of samples each particle performs before storing the sub run.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: Array with all params, array with all data, array with all log probabilities
        TODO check: are those really log probabilities?

    """

    data_stdevs = calc_kernel_width(data)
    sampling_dim = slice.shape[0]
    central_param = model.central_param

    # Initialize each walker at a Gaussian-drawn random, slightly different parameter close to the central parameter.
    # TODO Make random variation of initial walker positions dependent on sampling limits?
    initial_walker_positions = central_param[slice] + 0.002 * (
        np.random.rand(num_walkers, sampling_dim) - 0.5
    )

    # Count and print how many runs have already been performed for this model
    num_existing_files = result_manager.count_emcee_sub_runs(slice)
    logger.debug(f"{num_existing_files} existing files found")

    # Loop over the remaining sub runs and contiune the counter where it ended.
    for run in range(num_existing_files, num_existing_files + num_runs):
        logger.info(f"Run {run} of {num_runs}")

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
        sampler_results, final_walker_positions = run_emcee_once(
            model,
            data,
            data_stdevs,
            slice,
            initial_walker_positions,
            num_walkers,
            num_steps,
            num_processes,
        )

        result_manager.save_run(
            model, slice, run, sampler_results, final_walker_positions
        )

    (
        overall_params,
        overall_sim_results,
        overall_density_evals,
    ) = concatenate_emcee_sampling_results(model, result_manager, slice)
    result_manager.save_overall(
        slice, overall_params, overall_sim_results, overall_density_evals
    )
    return overall_params, overall_sim_results, overall_density_evals


# TODO: Make this a method of the ResultManager? It uses the ResultManager to load the results and many hard coded paths.
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
    num_existing_files = result_manager.count_emcee_sub_runs(slice)
    logger.info(f"{num_existing_files} existing files found for concatenation")

    densityFiles = (
        result_manager.get_slice_path(slice)
        + "/DensityEvals/density_evals_{}.csv"
    )
    sim_result_files = (
        result_manager.get_slice_path(slice) + "/SimResults/sim_results_{}.csv"
    )
    paramFiles = result_manager.get_slice_path(slice) + "/Params/params_{}.csv"

    overall_params = np.vstack(
        [
            np.loadtxt(paramFiles.format(i), delimiter=",", ndmin=2)
            for i in range(num_existing_files)
        ]
    )
    overall_sim_results = np.vstack(
        [
            np.loadtxt(sim_result_files.format(i), delimiter=",", ndmin=2)
            for i in range(num_existing_files)
        ]
    )
    overall_density_evals = np.hstack(
        [
            np.loadtxt(densityFiles.format(i), delimiter=",")
            for i in range(num_existing_files)
        ]
    )

    return overall_params, overall_sim_results, overall_density_evals


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
        result_manager.get_slice_path(slice) + "/overall_params.csv",
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


def inference_mcmc(
    model: Model,
    data: np.ndarray,
    result_manager: ResultManager,
    slices: list[np.ndarray],
    num_processes: int,
    num_runs: int = 2,
    num_walkers: int = 10,
    num_steps: int = 2500,
    calc_walker_acceptance_bool: bool = False,
) -> typing.Tuple[
    typing.Dict[str, np.ndarray],
    typing.Dict[str, np.ndarray],
    typing.Dict[str, np.ndarray],
    ResultManager,
]:
    """This function runs a MCMC sampling for the given model and data.

    Args:
        model (Model): The model describing the mapping from parameters to data.
        data (np.ndarray): The data to be used for the inference.
        result_manager (ResultManager): The result manager to be used for the inference.
        slices (np.ndarray): A list of slices to be used for the inference.
        num_processes (int): The number of processes to be used for the inference.
        num_runs (int, optional): The number of runs to be used for the inference. Defaults to 2.
        num_walkers (int, optional): The number of walkers to be used for the inference. Defaults to 10.
        num_steps (int, optional): The number of steps to be used for the inference. Defaults to 2500.
        calc_walker_acceptance_bool (bool, optional): If True, the acceptance rate of the walkers is calculated and printed. Defaults to False.

    Returns:
        Tuple[typing.Dict[str, np.ndarray], typing.Dict[str, np.ndarray], typing.Dict[str, np.ndarray], ResultManager]: The parameter samples, the corresponding simulation results, the corresponding density
        evaluations for each slice and the result manager used for the inference.

    """
    # create the return dictionaries
    overall_params, overall_sim_results, overall_density_evals = {}, {}, {}

    for slice in slices:
        slice_name = result_manager.get_slice_name(slice)
        (
            overall_params[slice_name],
            overall_sim_results[slice_name],
            overall_density_evals[slice_name],
        ) = run_emcee_sampling(
            model=model,
            data=data,
            slice=slice,
            result_manager=result_manager,
            num_runs=num_runs,
            num_walkers=num_walkers,
            num_steps=num_steps,
            num_processes=num_processes,
        )

        if calc_walker_acceptance_bool:
            num_burn_in_steps = int(num_steps * 0.01)
            acceptance = calc_walker_acceptance(
                model, slice, num_walkers, num_burn_in_steps, result_manager
            )
            logger.info(f"Acceptance rate for slice {slice}: {acceptance}")

    return (
        overall_params,
        overall_sim_results,
        overall_density_evals,
        result_manager,
    )
