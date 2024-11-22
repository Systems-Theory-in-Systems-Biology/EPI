# TODO: Import Path from pathlib?
import json
import os
import shutil
from os import path
from typing import Dict, Optional, Tuple

import numpy as np
import seedir
from seedir import FakeDir, FakeFile

from eulerpi import logger
from eulerpi.core.inference_types import InferenceType
from eulerpi.core.models import BaseModel
from .path_gen import get_slice_name, get_run_path

class OutputWriter:
    """The output writer is responsible for saving the results of the inference.

    Attributes:
        model_name(str): The name of the model (e.g. "temperature"). It is used to create the folder structure.
        run_name(str): The name of the run which shall be saved. It is used to create subfolders for different runs.
    """

    def __init__(
        self, model_name: str, run_name: str
    ) -> None:
        """Creates an instance of the output writer for saving the results of the inference.

        Args:
            model_name (str): Name of the model.
            run_name (str): Name of the run.
        """
        self.model_name = model_name
        self.run_name = run_name 


    def create_output_folder_structure(self) -> None:
        """Creates the subfolders in `Output` for the given slice where all simulation results
        are stored for this model and slice. No files are deleted during this action.

        """
        outputFolderStructure = (
            "Output/ \n"
            "  - {modelName}/ \n"
            "    - {runName}/ \n"
            "       - DensityEvals/ \n"
            "       - Params/ \n"
            "       - SimResults/ \n"
        )
        path = "."
        structure = outputFolderStructure

        def create(f, root):
            """

            Args:
              f:
              root:

            Returns:

            """
            fpath = f.get_path()
            joined = os.path.join(root, fpath)
            if isinstance(f, FakeDir):
                try:
                    os.mkdir(joined)
                except FileExistsError:
                    logger.info(f"Directory `{joined}` already exists")
            elif isinstance(f, FakeFile):
                try:
                    with open(joined, "w"):
                        pass
                except FileExistsError:
                    logger.info(f"File `{joined}` already exists")

        fakeStructure = seedir.fakedir_fromstring(
            structure.format(
                modelName=self.model_name,
                runName=self.run_name,
            )
        )
        fakeStructure.realize = lambda path_arg: fakeStructure.walk_apply(
            create, root=path_arg
        )
        fakeStructure.realize(path)

    def delete_output_folder_structure(self) -> None:
        """Deletes the `Output` folder."""
        try:
            path = get_run_path(self.model_name, self.run_name)
            shutil.rmtree(path)
        except FileNotFoundError:
            logger.info(
                f"Folder structure for slice {slice} does not exist"
            )

    def count_emcee_sub_runs(self, slice: np.ndarray) -> int:
        """This data organization function counts how many sub runs are saved for the specified scenario.

        Args:
            slice(np.ndarray): The slice for which the number of sub runs should be counted.

        Returns:
          num_existing_files(int): The number of completed sub runs of the emcee particle swarm sampler.

        """
        # Initialize the number of existing files to be 0
        num_existing_files = 0

        # Increase the just defined number until no corresponding file is found anymore ...
        while path.isfile(
            self.get_slice_path(slice)
            + "/PushforwardEvals/"
            + "pushforward_evals_"
            + str(num_existing_files)
            + ".csv"
        ):
            num_existing_files += 1

        return num_existing_files
    
    def save_sampler_run(
        self,
        model: BaseModel,
        slice: np.ndarray,
        run,
        sampler_results: np.ndarray,
        final_walker_positions: np.ndarray,
    ) -> None:
        """Saves the results of a single run of the emcee particle swarm sampler.
        sampler_results has the shape (num_walkers * num_steps, sampling_dim + data_dim + 1), we save them
        as seperate files in the folders 'Params' and'PushforwardEvals' and 'DensityEvals'.

        Args:
            model(BaseModel): The model for which the results will be saved
            slice(np.ndarray): The slice for which the results will be saved
            run(int): The run for which the results will be saved
            sampler_results(np.ndarray): The results of the sampler, expects an np.array with shape (num_walkers * num_steps, sampling_dim + data_dim + 1)
            final_walker_positions(np.ndarray): The final positions of the walkers, expects an np.array with shape (num_walkers, sampling_dim)

        """

        sampling_dim = final_walker_positions.shape[1]

        results_path = get_run_path(self.model_name, self.run_name)

        # Save the parameters
        np.savetxt(
            results_path + "/Params/params_" + str(run) + ".csv",
            sampler_results[:, :sampling_dim],
            delimiter=",",
        )

        # Save the density evaluations
        np.savetxt(
            results_path + "/DensityEvals/density_evals_" + str(run) + ".csv",
            sampler_results[:, -1],
            delimiter=",",
        )

        # Save the simulation results
        np.savetxt(
            results_path + "/PushforwardEvals/pushforward_evals_" + str(run) + ".csv",
            sampler_results[:, sampling_dim : sampling_dim + model.data_dim],
            delimiter=",",
        )

        # Save the final walker positions
        np.savetxt(
            results_path + "/final_walker_positions_" + str(run) + ".csv",
            final_walker_positions,
            delimiter=",",
        )

    def save_grid_based_run(
        self, params, pushforward_evals, density_evals
    ):
        """Saves the results of a grid-based run.

        Args:
            params(np.ndarray): The parameters of the grid.
            pushforward_evals(np.ndarray): The pushforward of the points in the grid.
            density_evals(np.ndarray): The densities at the grid points.

        """
        # Save the three just-created files.
        np.savetxt(
            self.get_slice_path() + "/overall_density_evals.csv",
            density_evals,
            delimiter=",",
        )
        np.savetxt(
            self.get_slice_path() + "/overall_sim_results.csv",
            pushforward_evals,
            delimiter=",",
        )
        np.savetxt(
            self.get_slice_path() + "/overall_params.csv",
            params,
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
            "slice": get_slice_name(slice),
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
            get_run_path(self.model_name, self.run_name) + "/inference_information.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(information, file, ensure_ascii=False, indent=4)

    