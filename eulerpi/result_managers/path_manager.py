import os
import shutil

import numpy as np
import seedir
from seedir import FakeDir, FakeFile

from eulerpi import logger


class PathManager:

    def __init__(self, model_name: str, run_name):
        self.model_name = model_name
        self.run_name = run_name

    def get_slice_name(self, slice: np.ndarray) -> str:
        """This organization function returns a string name for a given slice.

        Args:
            slice(np.ndarray): The slice for which the name will be returned.

        Returns:
            str: The string name of the slice.

        """
        return "slice_" + "".join(["Q" + str(i) for i in slice])

    def get_output_path(self) -> str:
        """Returns the path to the output folder, containing also intermediate results.

        Returns:
            str: The path to the output folder, containing also intermediate results.

        """
        return os.path.join("Output", self.model_name)

    def get_run_path(self) -> str:
        """Returns the path to the folder where the results for the given run are stored.

        Returns:
            str: The path to the folder where the results for the given run are stored.

        """
        return os.path.join("Output", self.model_name, self.run_name)

    def create_output_folder_structure(self) -> None:
        """Creates the subfolders in `Output` for the given run where all simulation results
        are stored for this model and run. No files are deleted during this action.

        """
        outputFolderStructure = (
            "Output/ \n"
            "  - {modelName}/ \n"
            "    - {runName}/ \n"
            "       - DensityEvals/ \n"
            "       - Params/ \n"
            "       - PushforwardEvals/ \n"
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
            path = self.get_run_path()
            shutil.rmtree(path)
        except FileNotFoundError:
            logger.info(
                f"Folder structure for run_name {self.run_name} does not exist"
            )

    def count_emcee_sub_runs(self) -> int:
        """This data organization function counts how many sub runs are saved for the specified scenario.

        Returns:
          num_existing_files(int): The number of completed sub runs of the emcee particle swarm sampler.

        """
        # Initialize the number of existing files to be 0
        num_existing_files = 0

        # Increase the just defined number until no corresponding file is found anymore ...
        while os.path.isfile(
            self.get_run_path()
            + "/PushforwardEvals/"
            + "pushforward_evals_"
            + str(num_existing_files)
            + ".csv"
        ):
            num_existing_files += 1

        return num_existing_files

    def get_inference_information_path(self) -> str:
        return self.get_run_path() + "/inference_information.json"
