# TODO: Import Path from pathlib?
import json
import os
import shutil
from os import path
from typing import Optional, Tuple

import numpy as np
import seedir
from seedir import FakeDir, FakeFile

from eulerpi import logger
from eulerpi.core.inference_types import InferenceType
from eulerpi.core.model import Model


class ResultManager:
    """The result manager is responsible for saving the results of the inference and loading them again.

    Attributes:
        model_name(str): The name of the model (e.g. "temperature"). It is used to create the folder structure.
        run_name(str): The name of the run which shall be saved. It is used to create subfolders for different runs.
    """

    def __init__(self, model_name: str, run_name: str) -> None:
        self.model_name = model_name
        self.run_name = run_name

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

    def create_application_folder_structure(
        self, model: Model, slices: Optional[list[np.ndarray]] = None
    ) -> None:
        """Creates the `Application` folder including subfolder where all simulation results
        are stored for this model. No files are deleted during this action.

        Args:
            model(Model): The model for which the folder structure will be created
            slices(list[np.ndarray]): The slices for which the folder structure will be created

        """

        if slices is None:
            slices = [np.arange(model.param_dim)]
        for slice in slices:
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

    def delete_application_folder_structure(self, model, slices) -> None:
        """Deletes the `Applications` subfolder

        Args:
            slices(list[np.ndarray]): The slices for which the folder structure will be deleted
            model(Model): The model for which the folder structure will be deleted

        """
        for slice in slices:
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

    def save_run(
        self,
        model: Model,
        slice: np.ndarray,
        run,
        sampler_results: np.ndarray,
        final_walker_positions: np.ndarray,
    ) -> None:
        """Saves the results of a single run of the emcee particle swarm sampler.
        sampler_results has the shape (num_walkers * num_steps, sampling_dim + data_dim + 1), we save them
        as seperate files in the folders 'Params' and'SimResults' and 'DensityEvals'.

        Args:
            model(Model): The model for which the results will be saved
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
            sampler_results[
                :, sampling_dim : model.param_dim + model.data_dim
            ],
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
            slice(np.ndarray): The slice for which the results will be saved. TODO document dimensions of overall_params, overall_sim_results, overall_density_evals
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
        model: Model,
        inference_type: InferenceType,
        num_processes: int,
        **kwargs,
    ) -> None:
        """Saves the information about the inference run.

        Args:
            slice(np.ndarray): The slice for which the results will be saved.
            model(Model): The model for which the results will be saved.
            inference_type(InferenceType): The type of inference that was performed.
            num_processes(int): The number of processes that were used for the inference.
            **kwargs: Additional information about the inference run.
                num_runs(int): The number of runs that were performed. Only for mcmc inference.
                num_walkers(int): The number of walkers that were used. Only for mcmc inference.
                num_steps(int): The number of steps that were performed. Only for mcmc inference.
                num_burn_in_samples(int): The number of samples that were ignored. Only for mcmc inference.
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
            "slice": self.get_slice_name(slice),
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
            self.get_slice_path(slice) + "/inference_information.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(information, file, ensure_ascii=False, indent=4)

    def load_sim_results(
        self, slice: np.ndarray, num_burn_in_samples: int, occurrence: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load the files generated by the EPI algorithm through sampling

        Args:
            slice(np.ndarray): Slice for which the results will be loaded
            num_burn_in_samples(int): Ignore the first samples of each chain
            occurrence(int): step of sampling from chains

        Returns:
            typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: The density evaluations, the simulation results and the parameter chain

        """
        results_path = self.get_slice_path(slice)

        density_evals = np.loadtxt(
            results_path + "/overall_density_evals.csv",
            delimiter=",",
        )[num_burn_in_samples::occurrence]
        sim_results = np.loadtxt(
            results_path + "/overall_sim_results.csv",
            delimiter=",",
            ndmin=2,
        )[num_burn_in_samples::occurrence, :]
        param_chain = np.loadtxt(
            results_path + "/overall_params.csv",
            delimiter=",",
            ndmin=2,
        )[num_burn_in_samples::occurrence, :]
        return param_chain, sim_results, density_evals
