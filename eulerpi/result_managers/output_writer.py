import json
from typing import Optional

import numpy as np

from eulerpi.inference import InferenceType
from eulerpi.models import BaseModel

from .path_manager import PathManager


class OutputWriter:
    """The output writer is responsible for saving the results of the inference.

    Attributes:
        model_name(str): The name of the model (e.g. "temperature"). It is used to create the folder structure.
        run_name(str): The name of the run which shall be saved. It is used to create subfolders for different runs.
        path_manager(PathManager): A path manager Object to manage file paths.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        run_name: Optional[str] = None,
        path_manager: Optional[PathManager] = None,
    ) -> None:
        """Creates an instance of the output writer for saving the results of the inference.

        Args:
            model_name (str, optional): Name of the model. Defaults to None. Only required (and used) if no path manager is provided.
            run_name (str, optional): Name of the run. Defaults to None. Only required (and used) if no path manager is provided.
            path_manager (PathManager, optional): Path manager Object to manage the folder structure. Defaults to None creates a new path manager.
        """
        self.path_manager = path_manager or PathManager(model_name, run_name)
        self.model_name = self.path_manager.model_name
        self.run_name = self.path_manager.run_name

    def get_run_path(self) -> str:
        """Returns the path to the folder where the results for the given run are stored.

        Returns:
            str: The path to the folder where the results for the given run are stored.

        """
        return self.path_manager.get_run_path()

    def create_output_folder_structure(self) -> None:
        """Creates the subfolders in `Output` for the given run where all simulation results
        are stored for this model and run. No files are deleted during this action.

        """
        self.path_manager.create_output_folder_structure()

    def delete_output_folder_structure(self) -> None:
        """Deletes the `Output` folder."""
        self.path_manager.delete_output_folder_structure()

    def save_sampler_run(
        self,
        model: BaseModel,
        chain_number: int,
        sampler_results: np.ndarray,
        final_walker_positions: np.ndarray,
    ) -> None:
        """Saves the results of a single run of the emcee particle swarm sampler.
        sampler_results has the shape (num_walkers * num_steps, sampling_dim + data_dim + 1), we save them
        as seperate files in the folders 'Params' and'PushforwardEvals' and 'DensityEvals'.

        Args:
            model(BaseModel): The model for which the results will be saved
            slice(np.ndarray): The slice for which the results will be saved
            chain_number(int): The run for which the results will be saved
            sampler_results(np.ndarray): The results of the sampler, expects an np.array with shape (num_walkers * num_steps, sampling_dim + data_dim + 1)
            final_walker_positions(np.ndarray): The final positions of the walkers, expects an np.array with shape (num_walkers, sampling_dim)

        """

        sampling_dim = final_walker_positions.shape[1]

        results_path = self.get_run_path()

        # Save the parameters
        np.savetxt(
            results_path
            + f"/{self.path_manager.PARAMS_FOLDER}/raw_params_"
            + str(chain_number)
            + ".csv",
            sampler_results[:, :sampling_dim],
            delimiter=",",
        )

        # Save the density evaluations
        np.savetxt(
            results_path
            + f"/{self.path_manager.DENSITY_EVALS_FOLDER}/raw_density_evals_"
            + str(chain_number)
            + ".csv",
            sampler_results[:, -1],
            delimiter=",",
        )

        # Save the pushforward evaluations
        np.savetxt(
            results_path
            + f"/{self.path_manager.PUSHFORWARD_EVALS_FOLDER}/raw_pushforward_evals_"
            + str(chain_number)
            + ".csv",
            sampler_results[:, sampling_dim : sampling_dim + model.data_dim],
            delimiter=",",
        )

        # Save the final walker positions
        np.savetxt(
            results_path
            + "/final_walker_positions_"
            + str(chain_number)
            + ".csv",
            final_walker_positions,
            delimiter=",",
        )

    def save_current_walker_pos(self, params: np.ndarray) -> None:
        """Saves the current walker positions.

        Args:
            params(np.ndarray): The current walker positions.

        """
        positions_path = self.path_manager.get_current_walker_position_path()
        np.savetxt(
            positions_path,
            params,
            delimiter=",",
        )

    def save_grid_based_run(
        self,
        params: np.ndarray,
        pushforward_evals: np.ndarray,
        density_evals: np.ndarray,
    ):
        """Saves the results of a grid-based run.

        Args:
            params(np.ndarray): The parameters of the grid.
            pushforward_evals(np.ndarray): The pushforward of the points in the grid.
            density_evals(np.ndarray): The densities at the grid points.

        """
        results_path = self.get_run_path()

        # Save the parameters
        np.savetxt(
            results_path + "/params.csv",
            params,
            delimiter=",",
        )
        np.savetxt(
            results_path + "/pushforward_evals.csv",
            pushforward_evals,
            delimiter=",",
        )
        np.savetxt(
            results_path + "/density_evals.csv",
            density_evals,
            delimiter=",",
        )

    def save_inference_information(
        self,
        slice: np.ndarray,
        model: BaseModel,
        inference_type: InferenceType,
        num_processes: int,
        **kwargs,
    ) -> None:
        """Saves the information about the inference run.

        Args:
            slice(np.ndarray): The slice for which the results will be saved.
            model(BaseModel): The model for which the results will be saved.
            inference_type(InferenceType): The type of inference that was performed.
            num_processes(int): The number of processes that were used for the inference.
            **kwargs: Additional information about the inference run.
                num_runs(int): The number of runs that were performed. Only for mcmc inference.
                num_walkers(int): The number of walkers that were used. Only for mcmc inference.
                num_steps(int): The number of steps that were performed. Only for mcmc inference.
                num_burn_in_samples(int): Number of samples that will be ignored per chain (i.e. walker). Only for mcmc inference.
                thinning_factor(int): The thinning factor that was used to thin the Markov chain. Only for mcmc inference.
                load_balancing_safety_faktor(int): The safety factor that was used for load balancing. Only for dense grid inference.
                num_grid_points(np.ndarray): The number of grid points that were used. Only for dense grid inference.
                dense_grid_type(DenseGridType): The type of dense grid that was used: either equidistant or chebyshev. Only for dense grid inference.
                num_levels(int): The number of sparse grid levels that were used. Only for sparse grid inference.

        Raises:
            ValueError: If the inference type is unknown.

        """
        information = {
            "model": model.name,
            "slice": self.path_manager.get_slice_name(slice),
            "inference_type": inference_type.name,
            "num_processes": num_processes,
        }
        information.update(dict(kwargs))
        if "num_grid_points" in information:
            information["num_grid_points"] = np.array2string(
                information["num_grid_points"]
            )
        if "dense_grid_type" in information:
            information["dense_grid_type"] = information[
                "dense_grid_type"
            ].name
        # save information as json file
        with open(
            self.path_manager.get_inference_information_path(),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(information, file, ensure_ascii=False, indent=4)

    def append_inference_information(
        self,
        **kwargs,
    ) -> None:
        """Appends information about the inference run to the existing information."""
        pass  # TODO implement
