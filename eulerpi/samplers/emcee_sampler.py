import multiprocessing
import warnings
from typing import Callable, Tuple

import emcee
import numpy as np

from .sampler import Sampler


class EmceeSampler(Sampler):
    def __init__(
        self,
        sampling_dim: int,
        logger,
    ):
        self.sampling_dim = sampling_dim
        self.logger = logger

    def run(
        self,
        logdensity_blob_function: Callable,
        initial_walker_positions: np.ndarray,
        num_walkers: int,
        num_steps: int,
        num_processes: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pool = multiprocessing.get_context("spawn").Pool(
            processes=num_processes
        )
        try:
            emcee_sampler = emcee.EnsembleSampler(
                num_walkers,
                self.sampling_dim,
                logdensity_blob_function,
                pool=pool,
            )
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
                final_walker_positions, _, _, _ = emcee_sampler.run_mcmc(
                    initial_walker_positions,
                    num_steps,
                    tune=True,
                    progress=True,
                )
        except ValueError as e:
            if "Probability function returned NaN" in str(e):
                raise ValueError(
                    "Probability function returned NaN. "
                    "You possibly have to exclude data dimensions which do not depend on the paramaters. "
                    "In addition your parameters should not be linearly dependent."
                )
            else:
                raise e
        finally:
            pool.close()
            pool.join()

        sampler_results: np.ndarray = emcee_sampler.get_blobs()
        sampler_results = sampler_results.reshape(num_steps * num_walkers, -1)

        self.logger.info(
            f"The acceptance fractions of the emcee sampler per walker are: {np.round(emcee_sampler.acceptance_fraction, 2)}"
        )
        try:
            corrTimes = emcee_sampler.get_autocorr_time()
            self.logger.info(f"autocorrelation time: {corrTimes[0]}")
        except emcee.autocorr.AutocorrError as e:
            self.logger.warning(
                "The autocorrelation time could not be calculate reliable"
            )
        return sampler_results, final_walker_positions
