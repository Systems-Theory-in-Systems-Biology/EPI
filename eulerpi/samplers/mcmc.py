"""Module implementing the :py:func:`inference <eulerpi.inference.inference>` with a Markov Chain Monte Carlo(MCMC) sampling.

The inference with a sampling based approach approximates the the (joint/marginal) parameter distribution(s) by calculating a parameter Markov Chain
using multiple walkers in parallel. This module is currently based on the emcee package.

.. _emcee: https://emcee.readthedocs.io/en/stable/

.. note::

    The functions in this module are mainly intended for internal use and are accessed by :func:`inference <eulerpi.inference>` function.
    Read the documentation of :func:`inference_mcmc <inference_mcmc>` to learn more about the available options for the MCMC based inference.
"""

from multiprocessing import get_context

import emcee
import numpy as np

from eulerpi.data_transformations import DataTransformation
from eulerpi.evaluation.density import evaluate_log_density
from eulerpi.evaluation.kde import KDE
from eulerpi.logger import logger
from eulerpi.models import BaseModel

INFERENCE_NAME = "MCMC"


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
            evaluate_log_density,
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
