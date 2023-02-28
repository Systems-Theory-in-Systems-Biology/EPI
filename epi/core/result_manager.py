# TODO: Import Path from pathlib?
import os
import shutil
from os import path

import numpy as np
import seedir
from seedir import FakeDir, FakeFile

from epi import logger


class ResultManager:
    @staticmethod
    def countEmceeSubRuns(model) -> int:
        """This data organization function counts how many sub runs are saved for the specified scenario.

        :param model: The model for which the files will be counted
        :return: numExistingFiles (number of completed sub runs of the emcee particle swarm sampler)
        """
        # Initialize the number of existing files to be 0
        numExistingFiles = 0

        # Increase the just defined number until no corresponding file is found anymore ...
        while path.isfile(
            ResultManager.getApplicationPath(model)
            + "/DensityEvals/"
            + str(numExistingFiles)
            + ".csv"
        ):
            numExistingFiles += 1

        return numExistingFiles

    @staticmethod
    def getSliceName(slice: np.ndarray) -> str:
        return "Slice_" + "Q".join([str(i) for i in slice])

    @staticmethod
    def getSlicePath(self, slice: np.ndarray) -> str:
        sliceName = ResultManager.getSliceName(slice)
        return os.path.join("Applications", self.name, sliceName)

    @staticmethod
    def createApplicationFolderStructure(
        model, slices: list[np.ndarray] = None
    ) -> None:
        """Creates the `Application` folder including subfolder where all simulation results
        are stored for this model. No files are deleted during this action.
        """

        if slices is None:
            slices = [np.arange(model.paramDim)]
        for slice in slices:
            ResultManager.createSliceFolderStructure(model, slice)

    @staticmethod
    def createSliceFolderStructure(model, slice: np.ndarray) -> None:
        applicationFolderStructure = (
            "Applications/ \n"
            "  - {modelName}/ \n"
            "    - {sliceName}/ \n"
            "       - DensityEvals/ \n"
            "       - Params/ \n"
            "       - SimResults/ \n"
        )
        path = "."
        structure = applicationFolderStructure

        def create(f, root):
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

        sliceName = ResultManager.getSliceName(slice)
        fakeStructure = seedir.fakedir_fromstring(
            structure.format(modelName=model.name, sliceName=sliceName)
        )
        fakeStructure.realize = lambda path_arg: fakeStructure.walk_apply(
            create, root=path_arg
        )
        fakeStructure.realize(path)

    @staticmethod
    def deleteApplicationFolderStructure(model, slices) -> None:
        """Deletes the models `Applications` subfolder"""
        for slice in slices:
            try:
                ResultManager.deleteSliceFolderStructure(model, slice)
            except FileNotFoundError:
                logger.info(
                    f"Folder structure for slice {slice} does not exist"
                )

    @staticmethod
    def deleteSliceFolderStructure(model, slice: np.ndarray) -> None:
        path = ResultManager.getSlicePath(model, slice)
        shutil.rmtree(path)

    @staticmethod
    def getApplicationPath(model) -> str:
        """Returns the path to the simulation results folder, containing also intermediate results

        :return: path as string to the simulation folder
        :rtype: str
        """
        path = "Applications/" + model.name
        return path

    @staticmethod
    def save_run(model, slice, run, samplerResults, finalWalkerPositions):
        """samplerResults has the shape (numWalkers * numSteps, samplingDim + dataDim + 1)
        we save them as seperate files in the folders 'Params' and'SimResults' and 'densityEvals'
        """

        samplingDim = finalWalkerPositions.shape[1]

        results_path = ResultManager.getApplicationPath(model)

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

    @staticmethod
    def save_overall(
        model, slice, overallParams, overallSimResults, overallDensityEvals
    ):
        # Save the three just-created files.
        np.savetxt(
            model.getApplicationPath() + "/OverallDensityEvals.csv",
            overallDensityEvals,
            delimiter=",",
        )
        np.savetxt(
            model.getApplicationPath() + "/OverallSimResults.csv",
            overallSimResults,
            delimiter=",",
        )
        np.savetxt(
            model.getApplicationPath() + "/OverallParams.csv",
            overallParams,
            delimiter=",",
        )

    @staticmethod
    def loadSimResults(model, numBurnSamples: int, occurrence: int):
        """Load the files generated by the EPI algorithm through sampling

        :param model: Model from which the results will be loaded
        :type model: Model
        :param numBurnSamples: Ignore the first samples of each chain
        :type numBurnSamples: int
        :param occurrence: step of sampling from chains
        :type occurrence: int
        :return: _description_
        :rtype: _type_
        """
        results_path = ResultManager.getApplicationPath(model)

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
