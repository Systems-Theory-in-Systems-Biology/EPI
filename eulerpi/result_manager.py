# TODO: Import Path from pathlib?
import json
import os
import shutil
from os import path
from typing import Dict, Optional, Tuple

import numpy as np
import seedir
from seedir import FakeDir, FakeFile

from eulerpi.logger import logger
from eulerpi.models import BaseModel


class ResultManager:
    """The result manager is responsible for saving the results of the inference and loading them again.

    Attributes:
        model_name(str): The name of the model (e.g. "temperature"). It is used to create the folder structure.
        run_name(str): The name of the run which shall be saved. It is used to create subfolders for different runs.
    """

    def __init__(
        self, model_name: str, run_name: str, slices=list[np.ndarray]
    ) -> None:
        self.model_name = model_name
        self.run_name = run_name
        self.slices = slices

    def count_sub_runs(self, slice: np.ndarray) -> int:
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
            + "/SimResults/"
            + "sim_results_"
            + str(num_existing_files)
            + ".csv"
        ):
            num_existing_files += 1

        return num_existing_files

    def get_slice_name(self, slice: np.ndarray) -> str:
        """This organization function returns the name of the folder for the current slice.

        Args:
            slice(np.ndarray): The slice for which the name of the folder will be returned.

        Returns:
            str: The name of the folder for the current slice.

        """

        return "Slice_" + "".join(["Q" + str(i) for i in slice])

    def get_slice_path(self, slice: np.ndarray) -> str:
        """Returns the path to the folder where the results for the given slice are stored.

        Args:
            slice(np.ndarray): The slice for which the path will be returned.

        Returns:
            str: The path to the folder where the results for the given slice are stored.

        """
        sliceName = self.get_slice_name(slice)
        return os.path.join(
            "Applications", self.model_name, self.run_name, sliceName
        )

    def create_application_folder_structure(self) -> None:
        """Creates the `Application` folder including subfolder where all simulation results
        are stored for this model. No files are deleted during this action.

        """

        for slice in self.slices:
            self.create_slice_folder_structure(slice)

    def create_slice_folder_structure(self, slice: np.ndarray) -> None:
        """Creates the subfolders in `Aplication` for the given slice where all simulation results
        are stored for this model and slice. No files are deleted during this action.

        Args:
            slice(np.ndarray): The slice for which the folder structure will be created

        """
        applicationFolderStructure = (
            "Applications/ \n"
            "  - {modelName}/ \n"
            "    - {runName}/ \n"
            "       - {sliceName}/ \n"
            "           - DensityEvals/ \n"
            "           - Params/ \n"
            "           - SimResults/ \n"
        )
        path = "."
        structure = applicationFolderStructure

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

        sliceName = self.get_slice_name(slice)
        fakeStructure = seedir.fakedir_fromstring(
            structure.format(
                modelName=self.model_name,
                runName=self.run_name,
                sliceName=sliceName,
            )
        )
        fakeStructure.realize = lambda path_arg: fakeStructure.walk_apply(
            create, root=path_arg
        )
        fakeStructure.realize(path)

    def delete_application_folder_structure(self) -> None:
        """Deletes the `Applications` subfolder"""
        for slice in self.slices:
            try:
                self.delete_slice_folder_structure(slice)
            except FileNotFoundError:
                logger.info(
                    f"Folder structure for slice {slice} does not exist"
                )

    def delete_slice_folder_structure(self, slice: np.ndarray) -> None:
        """Deletes the `Applications/[slice]` subfolder

        Args:
            slice(np.ndarray): The slice for which the folder structure will be deleted

        """
        path = self.get_slice_path(slice)
        shutil.rmtree(path)

    def get_application_path(self) -> str:
        """Returns the path to the simulation results folder, containing also intermediate results.

        Returns:
            str: The path to the simulation results folder, containing also intermediate results.

        """
        path = "Applications/" + self.model_name
        return path

    def save_subrun(
        self,
        model: BaseModel,
        slice: np.ndarray,
        run,
        sampler_results: np.ndarray,
        final_walker_positions: np.ndarray,
    ) -> None:
        """Saves the results of a single sub run of the emcee particle swarm sampler.
        sampler_results has the shape (num_walkers * num_steps, sampling_dim + data_dim + 1), we save them
        as seperate files in the folders 'Params' and'SimResults' and 'DensityEvals'.

        Args:
            model(BaseModel): The model for which the results will be saved
            slice(np.ndarray): The slice for which the results will be saved
            run(int): The run for which the results will be saved
            sampler_results(np.ndarray): The results of the sampler, expects an np.array with shape (num_walkers * num_steps, sampling_dim + data_dim + 1)
            final_walker_positions(np.ndarray): The final positions of the walkers, expects an np.array with shape (num_walkers, sampling_dim)

        """

        sampling_dim = final_walker_positions.shape[1]

        results_path = self.get_slice_path(slice)

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
            results_path + "/SimResults/sim_results_" + str(run) + ".csv",
            sampler_results[:, sampling_dim : sampling_dim + model.data_dim],
            delimiter=",",
        )

        # Save the final walker positions
        np.savetxt(
            results_path + "/final_walker_positions_" + str(run) + ".csv",
            final_walker_positions,
            delimiter=",",
        )

    def save_overall(
        self, slice, overall_params, overall_sim_results, overall_density_evals
    ):
        """Saves the results of all runs of the emcee particle swarm sampler for the given slice.

        Args:
            slice(np.ndarray): The slice for which the results will be saved. # TODO document dimensions of overall_params, overall_sim_results, overall_density_evals
            overall_params(np.ndarray): The results of the sampler.
            overall_sim_results(np.ndarray): The results of the sampler.
            overall_density_evals(np.ndarray): The results of the sampler.

        """
        # Save the three just-created files.
        np.savetxt(
            self.get_slice_path(slice) + "/overall_density_evals.csv",
            overall_density_evals,
            delimiter=",",
        )
        np.savetxt(
            self.get_slice_path(slice) + "/overall_sim_results.csv",
            overall_sim_results,
            delimiter=",",
        )
        np.savetxt(
            self.get_slice_path(slice) + "/overall_params.csv",
            overall_params,
            delimiter=",",
        )

    def save_inference_information(
        self,
        slice: np.ndarray,
        model: BaseModel,
        inference_type: str,
        num_processes: int,
        **kwargs,
    ) -> None:
        """Saves the information about the inference run.

        Args:
            slice(np.ndarray): The slice for which the results will be saved.
            model(BaseModel): The model for which the results will be saved.
            inference_type(str): The type of inference that was performed.
            num_processes(int): The number of processes that were used for the inference.
            **kwargs: Additional information about the inference run.
                num_runs(int): The number of runs that were performed. Only for mcmc inference.
                num_walkers(int): The number of walkers that were used. Only for mcmc inference.
                num_steps(int): The number of steps that were performed. Only for mcmc inference.
                num_burn_in_samples(int): Number of samples that will be ignored per chain (i.e. walker). Only for mcmc inference.
                thinning_factor(int): The thinning factor that was used to thin the Markov chain. Only for mcmc inference.
                load_balancing_safety_faktor(int): The safety factor that was used for load balancing. Only for dense grid inference.
                num_grid_points(np.ndarray): The number of grid points that were used. Only for dense grid inference.
                grid_type(GridType): The type of grid that was used: either equidistant, chebyshev, or sparse
                num_levels(int): The number of sparse grid levels that were used. Only for sparse grid inference.

        Raises:
            ValueError: If the inference type is unknown.

        """
        information = {
            "model": model.name,
            "slice": self.get_slice_name(slice),
            "inference_type": inference_type,
            "num_processes": num_processes,
        }
        information.update(dict(kwargs))
        if "num_grid_points" in information:
            information["num_grid_points"] = str(
                information["num_grid_points"]
            )

        # save information as json file
        with open(
            self.get_slice_path(slice) + "/inference_information.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(information, file, ensure_ascii=False, indent=4)

    def append_inference_information(self, slice, **kwargs):
        file_path = self.get_slice_path(slice) + "/inference_information.json"

        with open(file_path, "r") as file:
            inference_information = json.load(file)
        with open(file_path, "w") as file:
            inference_information.update(dict(kwargs))
            json.dump(
                inference_information, file, ensure_ascii=False, indent=4
            )

    def load_inference_results(
        self,
        slices: Optional[list[np.ndarray]] = None,
        num_burn_in_samples: Optional[int] = None,
        thinning_factor: Optional[int] = None,
    ) -> Tuple[
        Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]
    ]:
        """Load the inference results generated by EPI.

        Args:
            slices(list[np.ndarray]): Slices for which the results will be loaded. Default is None and loads all slices.
            num_burn_in_samples(int): Number of samples that will be ignored per chain (i.e. walker). Only for mcmc inference. Default is None and uses the value that was used for the inference stored in inference_information.json.
            thinning_factor(int): Thinning factor that will be used to thin the Markov chain. Only for mcmc inference. Default is None and uses the value that was used for the inference stored in each inference_information.json.

        Returns:
            typing.Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]: The parameters, the simulation results and the density evaluations.

        """
        if slices is None:
            slices = self.slices
        params = {}
        sim_results = {}
        density_evals = {}
        for slice in slices:
            slice_name = self.get_slice_name(slice)
            (
                params[slice_name],
                sim_results[slice_name],
                density_evals[slice_name],
            ) = self.load_slice_inference_results(
                slice, num_burn_in_samples, thinning_factor
            )
        return params, sim_results, density_evals

    def load_slice_inference_results(
        self,
        slice: np.ndarray,
        num_burn_in_samples: Optional[int] = None,
        thinning_factor: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load the files generated by the EPI algorithm through sampling

        Args:
            slice(np.ndarray): Slice for which the results will be loaded
            num_burn_in_samples(int): Number of samples that will be ignored per chain (i.e. walker). Only for mcmc inference. Default is None and uses the value that was used for the inference stored in inference_information.json.
            thinning_factor(int): Thinning factor that will be used to thin the Markov chain. Only for mcmc inference. Default is None and uses the value that was used for the inference stored in inference_information.json.

        Returns:
            typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: The parameters, the simulation results and the density evaluations.

        """
        results_path = self.get_slice_path(slice)

        # load information from json file
        with open(results_path + "/inference_information.json", "r") as file:
            inference_information = json.load(file)

        # grid inference: load directly from overall results
        if (
            inference_information["inference_type"] == "GRID"
        ):  # TODO: Specialice result manager for the Different samplings? Depending on the details here is bad
            if num_burn_in_samples is not None:
                logger.info(
                    f"For inference type {inference_information['inference_type']}, num_burn_in_samples is ignored."
                )
            if thinning_factor is not None:
                logger.info(
                    f"For inference type {inference_information['inference_type']}, thinning_factor is ignored."
                )
            overall_density_evals = np.loadtxt(
                results_path + "/overall_density_evals.csv",
                delimiter=",",
            )
            overall_sim_results = np.loadtxt(
                results_path + "/overall_sim_results.csv",
                delimiter=",",
                ndmin=2,
            )
            overall_param_chain = np.loadtxt(
                results_path + "/overall_params.csv",
                delimiter=",",
                ndmin=2,
            )
            return (
                overall_param_chain,
                overall_sim_results,
                overall_density_evals,
            )

        # MCMC inference
        # use default values saved in inference_information if not specified
        if num_burn_in_samples is None:
            num_burn_in_samples = inference_information["num_burn_in_samples"]

        if thinning_factor is None:
            thinning_factor = inference_information["thinning_factor"]

        num_steps = inference_information["num_steps"]
        num_walkers = inference_information["num_walkers"]

        # load samples from raw chains
        for i in range(inference_information["num_runs"]):
            density_evals = np.loadtxt(
                results_path + f"/DensityEvals/density_evals_{i}.csv",
                delimiter=",",
            )
            sim_results = np.loadtxt(
                results_path + f"/SimResults/sim_results_{i}.csv",
                delimiter=",",
                ndmin=2,
            )
            params = np.loadtxt(
                results_path + f"/Params/params_{i}.csv",
                delimiter=",",
                ndmin=2,
            )
            if i == 0:
                param_dim = params.shape[1]
                data_dim = sim_results.shape[1]
                overall_density_evals = density_evals.reshape(
                    num_steps, num_walkers, 1
                )
                overall_sim_results = sim_results.reshape(
                    num_steps, num_walkers, data_dim
                )
                overall_params = params.reshape(
                    num_steps, num_walkers, param_dim
                )
            else:
                overall_density_evals = np.concatenate(
                    (
                        overall_density_evals,
                        density_evals.reshape(num_steps, num_walkers, 1),
                    )
                )
                overall_sim_results = np.concatenate(
                    (
                        overall_sim_results,
                        sim_results.reshape(num_steps, num_walkers, data_dim),
                    )
                )
                overall_params = np.concatenate(
                    (
                        overall_params,
                        params.reshape(num_steps, num_walkers, param_dim),
                    )
                )

        # thin and burn in
        return (
            overall_params[num_burn_in_samples::thinning_factor, :, :].reshape(
                -1, param_dim
            ),
            overall_sim_results[
                num_burn_in_samples::thinning_factor, :, :
            ].reshape(-1, data_dim),
            overall_density_evals[
                num_burn_in_samples::thinning_factor, :, :
            ].reshape(-1, 1),
        )
