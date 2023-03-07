# TODO: Import Path from pathlib?
import os
import shutil
import typing
from os import path

import numpy as np
import seedir
from seedir import FakeDir, FakeFile

from epi import logger
from epi.core.model import Model


class ResultManager:
    """ """

    def __init__(self, model_name, run_name) -> None:
        # TODO comment
        self.model_name = model_name
        self.run_name = run_name

    def countEmceeSubRuns(self, slice: np.ndarray) -> int:
        """This data organization function counts how many sub runs are saved for the specified scenario.

        Args:
            slice(np.ndarray): The slice for which the number of sub runs should be counted.

        Returns:
          int: numExistingFiles, the number of completed sub runs of the emcee particle swarm sampler.

        """
        # Initialize the number of existing files to be 0
        numExistingFiles = 0

        # Increase the just defined number until no corresponding file is found anymore ...
        while path.isfile(
            self.getSlicePath(slice)
            + "/SimResults/"
            + "simResults_"
            + str(numExistingFiles)
            + ".csv"
        ):
            numExistingFiles += 1

        return numExistingFiles

    def getSliceName(self, slice: np.ndarray) -> str:
        """This organization function returns the name of the folder for the current slice.

        Args:
            slice(np.ndarray): The slice for which the name of the folder will be returned.

        Returns:
            str: The name of the folder for the current slice.

        """

        return "Slice_" + "Q".join([str(i) for i in slice])

    def getSlicePath(self, slice: np.ndarray) -> str:
        """Returns the path to the folder where the results for the given slice are stored.

        Args:
            slice(np.ndarray): The slice for which the path will be returned.

        Returns:
            str: The path to the folder where the results for the given slice are stored.

        """
        sliceName = self.getSliceName(slice)
        return os.path.join(
            "Applications", self.model_name, self.run_name, sliceName
        )

    def createApplicationFolderStructure(
        self, model: Model, slices: list[np.ndarray] = None
    ) -> None:
        """Creates the `Application` folder including subfolder where all simulation results
        are stored for this model. No files are deleted during this action.

        Args:
            model(Model): The model for which the folder structure will be created
            slices(list[np.ndarray]): The slices for which the folder structure will be created

        """

        if slices is None:
            slices = [np.arange(model.paramDim)]
        for slice in slices:
            self.createSliceFolderStructure(slice)

    def createSliceFolderStructure(self, slice: np.ndarray) -> None:
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

        sliceName = self.getSliceName(slice)
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

    def deleteApplicationFolderStructure(self, model, slices) -> None:
        """Deletes the `Applications` subfolder

        Args:
            slices(list[np.ndarray]): The slices for which the folder structure will be deleted

        """
        for slice in slices:
            try:
                self.deleteSliceFolderStructure(slice)
            except FileNotFoundError:
                logger.info(
                    f"Folder structure for slice {slice} does not exist"
                )

    def deleteSliceFolderStructure(self, slice: np.ndarray) -> None:
        """Deletes the `Applications/[slice]` subfolder

        Args:
            slice(np.ndarray): The slice for which the folder structure will be deleted

        """
        path = self.getSlicePath(slice)
        shutil.rmtree(path)

    def getApplicationPath(self) -> str:
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
        samplerResults: np.ndarray,
        finalWalkerPositions: np.ndarray,
    ) -> None:
        """Saves the results of a single run of the emcee particle swarm sampler.
        SamplerResults has the shape (numWalkers * numSteps, samplingDim + dataDim + 1), we save them
        as seperate files in the folders 'Params' and'SimResults' and 'densityEvals'.

        Args:
            model(Model): The model for which the results will be saved
            slice(np.ndarray): The slice for which the results will be saved
            run(int): The run for which the results will be saved
            samplerResults(np.ndarray): The results of the sampler, expects an np.array with shape (numWalkers * numSteps, samplingDim + dataDim + 1)
            finalWalkerPositions(np.ndarray): The final positions of the walkers, expects an np.array with shape (numWalkers, samplingDim)

        """

        samplingDim = finalWalkerPositions.shape[1]

        results_path = self.getSlicePath(slice)

        # Save the parameters
        np.savetxt(
            results_path + "/Params/params_" + str(run) + ".csv",
            samplerResults[:, :samplingDim],
            delimiter=",",
        )

        # Save the density evaluations
        np.savetxt(
            results_path + "/DensityEvals/densityEvals_" + str(run) + ".csv",
            samplerResults[:, -1],
            delimiter=",",
        )

        # Save the simulation results
        np.savetxt(
            results_path + "/SimResults/simResults_" + str(run) + ".csv",
            samplerResults[:, samplingDim : model.paramDim + model.dataDim],
            delimiter=",",
        )

        # Save the final walker positions
        np.savetxt(
            results_path + "/finalWalkerPositions_" + str(run) + ".csv",
            finalWalkerPositions,
            delimiter=",",
        )

    def save_overall(
        self, slice, overallParams, overallSimResults, overallDensityEvals
    ):
        """Saves the results of all runs of the emcee particle swarm sampler for the given slice.

        Args:
            slice(np.ndarray): The slice for which the results will be saved. TODO document dimensions of overallParams, overallSimResults, overallDensityEvals
            overallParams(np.ndarray): The results of the sampler.
            overallSimResults(np.ndarray): The results of the sampler.
            overallDensityEvals(np.ndarray): The results of the sampler.

        """
        # Save the three just-created files.
        np.savetxt(
            self.getSlicePath(slice) + "/OverallDensityEvals.csv",
            overallDensityEvals,
            delimiter=",",
        )
        np.savetxt(
            self.getSlicePath(slice) + "/OverallSimResults.csv",
            overallSimResults,
            delimiter=",",
        )
        np.savetxt(
            self.getSlicePath(slice) + "/OverallParams.csv",
            overallParams,
            delimiter=",",
        )

    def loadSimResults(
        self, slice: np.ndarray, numBurnSamples: int, occurrence: int
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load the files generated by the EPI algorithm through sampling

        Args:
            slice(np.ndarray): Slice for which the results will be loaded
            numBurnSamples(int): Ignore the first samples of each chain
            occurrence(int): step of sampling from chains

        Returns:
            typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: The density evaluations, the simulation results and the parameter chain

        """
        results_path = self.getSlicePath(slice)

        densityEvals = np.loadtxt(
            results_path + "/OverallDensityEvals.csv",
            delimiter=",",
        )[numBurnSamples::occurrence]
        simResults = np.loadtxt(
            results_path + "/OverallSimResults.csv",
            delimiter=",",
            ndmin=2,
        )[numBurnSamples::occurrence, :]
        paramChain = np.loadtxt(
            results_path + "/OverallParams.csv",
            delimiter=",",
            ndmin=2,
        )[numBurnSamples::occurrence, :]
        return densityEvals, simResults, paramChain
