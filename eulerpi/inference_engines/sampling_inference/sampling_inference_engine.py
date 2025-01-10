import logging
from functools import partial
from typing import Optional, Tuple

import numpy as np

from eulerpi.data_transformations.data_transformation import DataTransformation
from eulerpi.estimation.kde import KDE
from eulerpi.evaluation.transformation import evaluate_log_density
from eulerpi.models.base_model import BaseModel
from eulerpi.result_managers import OutputWriter, ResultReader
from eulerpi.samplers.emcee_sampler import EmceeSampler
from eulerpi.samplers.sampler import Sampler
from eulerpi.utils.function_wrappers import FunctionWithDimensions

from ..inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


def calc_walker_acceptance(
    model: BaseModel,
    num_walkers: int,
    num_burn_in_samples: int,
    result_reader: ResultReader,
):
    """Calculate the acceptance ratio for each individual walker of the emcee chain.
    This is especially important to find "zombie" walkers, that are never moving.

    Args:
        model (BaseModel): The model for which the acceptance ratio should be calculated
        num_walkers (int): number of walkers in the emcee chain.
        num_burn_in_samples(int): Number of samples that will be deleted (burned) per chain (i.e. walker).
        result_reader (ResultReader): ResultReader to load the results from

    Returns:
        np.ndarray: Array with the acceptance ratio for each walker

    """
    thinning_factor = 1
    params, _, _ = result_reader.load_inference_results(
        num_burn_in_samples=num_burn_in_samples,
        thinning_factor=thinning_factor,
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


def combined_evaluation(
    param: np.ndarray,
    model: BaseModel,
    data_transformation: DataTransformation,
    kde: KDE,
    slice: np.ndarray,
):
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


class SamplingInferenceEngine(InferenceEngine):
    def run(
        self,
        slice: np.ndarray,
        output_writer: OutputWriter,
        num_processes: int,
        sampler: Sampler = None,
        num_walkers: int = 10,
        num_steps_per_sub_run: int = 500,
        num_sub_runs: int = 5,
        num_burn_in_samples: Optional[int] = None,
        thinning_factor: Optional[int] = None,
        get_walker_acceptance: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a representative sample from the transformed parameter density using the emcee particle swarm sampler.
        Inital values are not stored in the chain and each file contains <num_steps> blocks of size num_walkers.

        Args:
            model (BaseModel): The model which will be sampled
            data_transformation (DataTransformation): The data transformation used to normalize the data.
            slice (np.ndarray): slice of the parameter space which will be sampled.
            output_writer (OutputWriter): OutputWriter used to store the results.
            num_processes (int): number of parallel threads.
            num_walkers (int): number of particles in the particle swarm sampler.
            num_steps_per_sub_run (int): number of samples each particle performs before storing the sub run.
            num_sub_runs (int): The number of times intermediate results are written to file. Defaults to 5. Use a higher value if your problem has long run-times to prevent data loss should something go wrong.
            num_burn_in_samples(int): Number of samples that will be deleted (burned) per chain (i.e. walker). Only for mcmc inference.
            thinning_factor (int): thinning factor for the samples.
            get_walker_acceptance (bool): If True, the acceptance rate of the walkers is calculated and printed. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Array with all params, array with all pushforward evaluations, array with all log probabilities

        """
        output_writer.append_inference_information(
            slice=slice,
            num_walkers=num_walkers,
            num_steps_per_sub_run=num_steps_per_sub_run,
            num_burn_in_samples=num_burn_in_samples,
            thinning_factor=thinning_factor,
            get_walker_acceptance=get_walker_acceptance,
        )

        logdensity_blob_function = get_LogDensityEvaluator(
            self.model, self.data_transformation, self.kde, slice
        )

        if num_burn_in_samples is None:
            num_burn_in_samples = int(num_steps_per_sub_run * 0.1)
        if thinning_factor is None:
            thinning_factor = 1

        sampling_dim = slice.shape[0]

        if sampler is None:
            sampler = EmceeSampler(sampling_dim=sampling_dim, logger=logger)

        result_reader = ResultReader(
            output_writer.model_name, output_writer.run_name
        )

        for intermediate_file_index in range(num_sub_runs):
            if intermediate_file_index == 0:
                if (
                    last_position := result_reader.load_sampler_position()
                    is not None
                ):
                    initial_walker_positions = last_position
                    logger.info(
                        f"Continue sampling from saved sampler position in {result_reader.path_manager.get_run_path()}"
                    )
                else:
                    # Initialize each walker at a Gaussian-drawn random, slightly different parameter close to the central parameter.
                    # compute element-wise min of the difference between the central parameter and the lower sampling limit and the difference between the central parameter and the upper sampling limit
                    d_min = np.minimum(
                        self.model.central_param
                        - self.model.param_limits[:, 0],
                        self.model.param_limits[:, 1]
                        - self.model.central_param,
                    )
                    initial_walker_positions = self.model.central_param[
                        slice
                    ] + d_min[slice] * (
                        np.random.rand(num_walkers, sampling_dim) - 0.5
                    )

            # Count and print how many runs have already been performed for this model
            num_existing_files = (
                result_reader.path_manager.count_emcee_sub_runs()
            )
            logger.debug(f"{num_existing_files} existing files found")

            # Run the sampler.
            logger.info(f"Starting sampler run {num_existing_files}")
            sampler_results, final_walker_positions = sampler.run(
                logdensity_blob_function,
                initial_walker_positions,
                num_walkers,
                num_steps_per_sub_run,
                num_processes,
            )

            output_writer.save_sampler_run(
                model=self.model,
                sub_run_index=num_existing_files,
                sampler_results=sampler_results,
                final_walker_positions=final_walker_positions,
            )
            output_writer.save_current_walker_pos(final_walker_positions)
            initial_walker_positions = final_walker_positions

        (
            overall_params,
            overall_sim_results,
            overall_density_evals,
        ) = result_reader.load_inference_results(
            num_burn_in_samples, thinning_factor
        )

        if get_walker_acceptance:
            acceptance = calc_walker_acceptance(
                model=self.model,
                slice=slice,
                num_walkers=num_walkers,
                num_burn_in_samples=num_burn_in_samples,
                result_reader=result_reader,
            )
            logger.info(f"Acceptance rate for slice {slice}: {acceptance}")

        return (
            overall_params,
            overall_sim_results,
            overall_density_evals,
        )
