# TODO: Move to plots.py. Currently here because plots.py is full at the moment. would find it there to work on it
from enum import Enum
from multiprocessing import Pool

# from pathos.multiprocessing import Pool
from os import path

import emcee
import numpy as np

from epic import logger
from epic.core.functions import evalLogTransformedDensity
from epic.core.model import Model


def countEmceeSubRuns(model: Model) -> int:
    """This data organization function counts how many sub runs are saved for the specified scenario.

    :param model: The model for which the files will be counted
    :return: numExistingFiles (number of completed sub runs of the emcee particle swarm sampler)
    """
    # Initialize the number of existing files to be 0
    numExistingFiles = 0

    # Increase the just defined number until no corresponding file is found anymore ...
    while path.isfile(
        model.getApplicationPath()
        + "/DensityEvals/"
        + str(numExistingFiles)
        + ".csv"
    ):
        numExistingFiles += 1

    return numExistingFiles


def runEmceeSampling(
    model: Model,
    numRuns: int = 2,
    numWalkers: int = 10,
    numSteps: int = 2500,
    numProcesses: int = 10,
) -> None:
    """Create a representative sample from the transformed parameter density using the emcee particle swarm sampler.
        Inital values are not stored in the chain and each file contains <numSteps> blocks of size numWalkers.

    :param model: The model which will be sampled
    :param numRuns: (number of stored sub runs)
    :param numWalkers: (number of particles in the particle swarm sampler)
    :param numSteps: (number of samples each particle performs before storing the sub run)
    :param numProcesses: (number of parallel threads)
    :return: None, except for stored files
    """

    # Load data, data standard deviations and model characteristics for the specified model.
    (
        paramDim,
        dataDim,
        numDataPoints,
        centralParam,
        data,
        dataStdevs,
    ) = model.dataLoader()

    # Initialize each walker at a Gaussian-drawn random, slightly different parameter close to the central parameter.
    walkerInitParams = centralParam + 0.002 * (
        np.random.rand(numWalkers, paramDim) - 0.5
    )

    # Count and print how many runs have already been performed for this model
    numExistingFiles = countEmceeSubRuns(model)
    logger.debug(f"{numExistingFiles} existing files found")

    # Loop over the remaining sub runs and contiune the counter where it ended.
    for run in range(numExistingFiles, numExistingFiles + numRuns):
        logger.info(f"Run {run} of {numRuns}")

        # If there are current walker positions defined by runs before this one, use them.
        position_path = model.getApplicationPath() + "/currentPos.csv"
        if path.isfile(position_path):
            walkerInitParams = np.loadtxt(
                position_path,
                delimiter=",",
                ndmin=2,
            )
            logger.info(
                f"Continue sampling from saved sampler position in {position_path}"
            )

        else:
            logger.info("Start sampling from start")

        # Create a pool of worker processes.
        pool = Pool(processes=numProcesses)

        # define a custom move policy
        movePolicy = [
            (emcee.moves.WalkMove(), 0.1),
            (emcee.moves.StretchMove(), 0.1),
            (
                emcee.moves.GaussianMove(
                    0.00001, mode="sequential", factor=None
                ),
                0.8,
            ),
        ]
        # movePolicy = [(emcee.moves.GaussianMove(0.00001, mode='sequential', factor=None), 1.0)]

        # Call the sampler for all parallel workers (possibly use arg moves = movePolicy)
        sampler = emcee.EnsembleSampler(
            numWalkers,
            paramDim,
            evalLogTransformedDensity,
            pool=pool,
            moves=movePolicy,
            args=[model, data, dataStdevs],
        )

        # Extract the final walker position and close the pool of worker processes.
        finalPos, _, _, _ = sampler.run_mcmc(
            walkerInitParams, numSteps, tune=True, progress=True
        )
        pool.close()
        pool.join()

        # Save the current walker positions as initial values for the next run.
        np.savetxt(
            position_path,
            finalPos,
            delimiter=",",
        )

        # Create a large container for all sampling results (sampled parameters, corresponding simulation results and parameter densities) and fill it using the emcee blob option.
        allRes = np.zeros((numWalkers * numSteps, paramDim + dataDim + 1))

        for i in range(numSteps):
            for j in range(numWalkers):
                allRes[i * numWalkers + j, :] = sampler.blobs[i][j]

        # Save all sampling results in .csv files.
        np.savetxt(
            "Applications/"
            + model.getModelName()
            + "/Params/"
            + str(run)
            + ".csv",
            allRes[:, 0:paramDim],
            delimiter=",",
        )
        np.savetxt(
            "Applications/"
            + model.getModelName()
            + "/SimResults/"
            + str(run)
            + ".csv",
            allRes[:, paramDim : paramDim + dataDim],
            delimiter=",",
        )
        np.savetxt(
            "Applications/"
            + model.getModelName()
            + "/DensityEvals/"
            + str(run)
            + ".csv",
            allRes[:, -1],
            delimiter=",",
        )

        logger.info(
            f"The acceptance fractions of the emcee sampler per walker are: {np.round(sampler.acceptance_fraction, 2)}"
        )

        # TODO: Catch error with expect?
        # Print the autocorrelation time (produces a so-far untreated runtime error if chains are too short)
        # logger.info(f"autocorrelation time: {sampler.get_autocorr_time()[0]}")


def concatenateEmceeSamplingResults(model: Model):
    """Concatenate many sub runs of the emcee sampler to create 3 large files for sampled parameters, corresponding simulation results and density evaluations.
        These files are later used for result visualization.

    Input: model
    Output: <none except for stored files>
    """

    # Load data, data standard deviations and model characteristics for the specified model.
    (
        paramDim,
        dataDim,
        numDataPoints,
        centralParam,
        data,
        dataStdevs,
    ) = model.dataLoader()

    # Count and print how many sub runs are ready to be merged.
    numExistingFiles = countEmceeSubRuns(model)
    logger.info(f"{numExistingFiles} existing files found for concatenation")

    # Load one example file and use it to extract how many samples are stored per file.
    numSamplesPerFile = np.loadtxt(
        model.getApplicationPath() + "/Params/0.csv", delimiter=","
    ).shape[0]

    # The overall number of sampled is the number of sub runs multiplied with the number of samples per file.
    numSamples = numExistingFiles * numSamplesPerFile

    # Create containers large enough to store all sampling information.
    overallDensityEvals = np.zeros(numSamples)
    overallSimResults = np.zeros((numSamples, dataDim))
    overallParams = np.zeros((numSamples, paramDim))

    # Loop over all sub runs, load the respective sample files and store them at their respective places in the overall containers.
    for i in range(numExistingFiles):
        overallDensityEvals[
            i * numSamplesPerFile : (i + 1) * numSamplesPerFile
        ] = np.loadtxt(
            "Applications/"
            + model.getModelName()
            + "/DensityEvals/"
            + str(i)
            + ".csv",
            delimiter=",",
        )
        overallSimResults[
            i * numSamplesPerFile : (i + 1) * numSamplesPerFile, :
        ] = np.loadtxt(
            "Applications/"
            + model.getModelName()
            + "/SimResults/"
            + str(i)
            + ".csv",
            delimiter=",",
            ndmin=2,
        )
        overallParams[
            i * numSamplesPerFile : (i + 1) * numSamplesPerFile, :
        ] = np.loadtxt(
            "Applications/"
            + model.getModelName()
            + "/Params/"
            + str(i)
            + ".csv",
            delimiter=",",
            ndmin=2,
        )

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


def calcWalkerAcceptance(model: Model, numBurnSamples: int, numWalkers: int):
    """Calculate the acceptance ratio for each individual walker of the emcee chain.
        This is especially important to find "zombie" walkers, that are never movig.

    Input: model
           numBurnSamples (integer number of ignored first samples of each chain)
           numWalkers (integer number of emcee walkers)

    Output: acceptanceRatios (np.array of size numWalkers)
    """

    # load the emcee parameter chain
    params = np.loadtxt(
        model.getApplicationPath() + "/OverallParams.csv",
        delimiter=",",
    )[numBurnSamples:, :]

    # calculate the number of steps each walker walked
    # subtract 1 because we count the steps between the parameters
    numSteps = int(params.shape[0] / numWalkers) - 1
    logger.info(f"Number of steps fo each walker = {numSteps}")

    # create storage to count the number of accepted steps for each counter
    numAcceptedSteps = np.zeros(numWalkers)

    for i in range(numWalkers):
        for j in range(numSteps):
            numAcceptedSteps[i] += 1 - np.all(
                params[i + j * numWalkers, :]
                == params[i + (j + 1) * numWalkers, :]
            )

    # calculate the acceptance ratio by dividing the number of accepted steps by the overall number of steps
    acceptanceRatios = numAcceptedSteps / numSteps

    return acceptanceRatios


def inference(model: Model, data_path=None):
    if data_path is None:
        data_path = "Data/" + model.getModelName() + "Data.csv"
    # TODO: Do something with the data path!!!
    runEmceeSampling(model)
    concatenateEmceeSamplingResults(model)
    overallSimResults = np.loadtxt(
        model.getApplicationPath() + "/OverallSimResults.csv",
        delimiter=",",
    )
    return overallSimResults


class PlotType(Enum):
    Spider = 0
    D1 = 1
    D2 = 2


def plot(model: Model, pType: PlotType):
    if pType == pType.Spider:
        # TODO: spyderplot
        pass
    elif pType == pType.D1:
        # TODO: 1d plot, select variable to plot!
        pass
    elif pType == pType.D2:
        # TODO: 2d surf, select variables to plot!
        pass
    else:
        raise ValueError(
            f"Unexpected PlotType value encountered. The value is {pType}."
        )
