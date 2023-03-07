from multiprocessing import Pool

# from pathos.multiprocessing import Pool
from os import path

import emcee
import numpy as np

from epi import logger
from epi.core.kde import calcKernelWidth
from epi.core.model import Model
from epi.core.result_manager import ResultManager
from epi.core.transformations import evalLogTransformedDensity

NUM_RUNS = 2
NUM_WALKERS = 10
NUM_STEPS = 2500
NUM_PROCESSES = 4

# TODO: This works on the blob
# Return the samples.
# return sampler.get_chain(discard=0, thin=1, flat=True)
# TODO: This stores the sample as 2d array in the format walker1_step1, walker2_step1, walker3_step1, walker1_step2, walker2_step2, walker3_step2, ...
# samplerResults = samplerBlob.reshape(
#     numWalkers * numSteps, samplingDim + dataDim + 1
# )


def runEmceeOnce(
    model: Model,
    data: np.ndarray,
    dataStdevs: np.ndarray,
    slice: np.ndarray,
    initialWalkerPositions: np.ndarray,
    numWalkers: int,
    numSteps: int,
    numProcesses: int,
) -> np.ndarray:
    """Run the emcee particle swarm sampler once.

    Args:
      model: The model which will be sampled
      dataDim: dimension of the data)
      data: data)
      dataStdevs: standard deviations of the data)
      slice: slice of the parameter space which will be sampled)
      initialWalkerPositions: initial parameter values for the walkers)
      numWalkers: number of particles in the particle swarm sampler)
      numSteps: number of samples each particle performs before storing the sub run)
      numProcesses: number of parallel threads)
      model: Model:
      data: np.ndarray:
      dataStdevs: np.ndarray:
      slice: np.ndarray:
      initialWalkerPositions: np.ndarray:
      numWalkers: int:
      numSteps: int:
      numProcesses: int:

    Returns:
      samples from the transformed parameter density)

    """
    # Create a pool of worker processes.
    pool = Pool(processes=numProcesses)
    # define a custom move policy
    movePolicy = [
        (emcee.moves.WalkMove(), 0.1),
        (emcee.moves.StretchMove(), 0.1),
        (
            emcee.moves.GaussianMove(0.00001, mode="sequential", factor=None),
            0.8,
        ),
    ]
    # movePolicy = [(emcee.moves.GaussianMove(0.00001, mode='sequential', factor=None), 1.0)]
    samplingDim = slice.shape[0]

    # Call the sampler for all parallel workers (possibly use arg moves = movePolicy)
    sampler = emcee.EnsembleSampler(
        numWalkers,
        samplingDim,
        evalLogTransformedDensity,
        pool=pool,
        moves=movePolicy,
        args=[model, data, dataStdevs, slice],
    )
    # Extract the final walker position and close the pool of worker processes.
    finalWalkerPositionsitions, _, _, _ = sampler.run_mcmc(
        initialWalkerPositions, numSteps, tune=True, progress=True
    )
    pool.close()
    pool.join()

    # TODO: Keep as 3d array?
    # Should have shape (numSteps, numWalkers, paramDim+dataDim+1)
    samplerResults = sampler.get_blobs()
    dataDim = data.shape[1]
    samplerResults = samplerResults.reshape(
        numSteps * numWalkers, samplingDim + dataDim + 1
    )
    samplerResults = samplerResults.reshape(
        numWalkers * numSteps, samplingDim + dataDim + 1
    )

    logger.info(
        f"The acceptance fractions of the emcee sampler per walker are: {np.round(sampler.acceptance_fraction, 2)}"
    )
    try:
        corrTimes = sampler.get_autocorr_time()
        logger.info(f"autocorrelation time: {corrTimes[0]}")
    except emcee.autocorr.AutocorrError as e:
        logger.warning(
            "The autocorrelation time could not be calculate reliable"
        )

    return samplerResults, finalWalkerPositionsitions


def runEmceeSampling(
    model: Model,
    data: np.ndarray,
    slice: np.ndarray,
    result_manager: ResultManager,
    numRuns: int = NUM_RUNS,
    numWalkers: int = NUM_WALKERS,
    numSteps: int = NUM_STEPS,
    numProcesses: int = NUM_PROCESSES,
) -> None:
    """Create a representative sample from the transformed parameter density using the emcee particle swarm sampler.
       Inital values are not stored in the chain and each file contains <numSteps> blocks of size numWalkers.

    Args:
      model: The model which will be sampled
      numRuns: number of stored sub runs)
      numWalkers: number of particles in the particle swarm sampler)
      numSteps: number of samples each particle performs before storing the sub run)
      numProcesses: number of parallel threads)
      model: Model:
      data: np.ndarray:
      slice: np.ndarray:
      result_manager: ResultManager:
      numRuns: int:  (Default value = NUM_RUNS)
      numWalkers: int:  (Default value = NUM_WALKERS)
      numSteps: int:  (Default value = NUM_STEPS)
      numProcesses: int:  (Default value = NUM_PROCESSES)

    Returns:
      None, except for stored files

    """

    dataStdevs = calcKernelWidth(data)
    samplingDim = slice.shape[0]
    centralParam = model.centralParam

    # Initialize each walker at a Gaussian-drawn random, slightly different parameter close to the central parameter.
    # TODO Make random variation of initial walker positions dependent on sampling limits?
    initialWalkerPositions = centralParam[slice] + 0.002 * (
        np.random.rand(numWalkers, samplingDim) - 0.5
    )

    # Count and print how many runs have already been performed for this model
    numExistingFiles = result_manager.countEmceeSubRuns(slice)
    logger.debug(f"{numExistingFiles} existing files found")

    # Loop over the remaining sub runs and contiune the counter where it ended.
    for run in range(numExistingFiles, numExistingFiles + numRuns):
        logger.info(f"Run {run} of {numRuns}")

        # If there are current walker positions defined by runs before this one, use them.
        positionPath = result_manager.getSlicePath(slice) + "/currentPos.csv"
        if path.isfile(positionPath):
            initialWalkerPositions = np.loadtxt(
                positionPath,
                delimiter=",",
                ndmin=2,
            )
            logger.info(
                f"Continue sampling from saved sampler position in {positionPath}"
            )

        # Run the sampler.
        samplerResults, finalWalkerPositions = runEmceeOnce(
            model,
            data,
            dataStdevs,
            slice,
            initialWalkerPositions,
            numWalkers,
            numSteps,
            numProcesses,
        )

        result_manager.save_run(
            model, slice, run, samplerResults, finalWalkerPositions
        )

    (
        overallParams,
        overallSimResults,
        overallDensityEvals,
    ) = concatenateEmceeSamplingResults(model, result_manager, slice)
    result_manager.save_overall(
        slice, overallParams, overallSimResults, overallDensityEvals
    )
    return overallParams, overallSimResults, overallDensityEvals


def concatenateEmceeSamplingResults(
    model: Model, result_manager: ResultManager, slice: np.ndarray
):
    """Concatenate many sub runs of the emcee sampler to create 3 large files for sampled parameters, corresponding simulation results and density evaluations.
        These files are later used for result visualization.

    Input: model
    Output: <none except for stored files>

    Args:
      model: Model:
      result_manager: ResultManager:
      slice: np.ndarray:

    Returns:

    """

    # Count and print how many sub runs are ready to be merged.
    numExistingFiles = result_manager.countEmceeSubRuns(slice)
    logger.info(f"{numExistingFiles} existing files found for concatenation")

    densityFiles = (
        result_manager.getSlicePath(slice)
        + "/DensityEvals/densityEvals_{}.csv"
    )
    simResultsFiles = (
        result_manager.getSlicePath(slice) + "/SimResults/simResults_{}.csv"
    )
    paramFiles = result_manager.getSlicePath(slice) + "/Params/params_{}.csv"

    overallParams = np.vstack(
        [
            np.loadtxt(paramFiles.format(i), delimiter=",", ndmin=2)
            for i in range(numExistingFiles)
        ]
    )
    overallSimResults = np.vstack(
        [
            np.loadtxt(simResultsFiles.format(i), delimiter=",", ndmin=2)
            for i in range(numExistingFiles)
        ]
    )
    overallDensityEvals = np.hstack(
        [
            np.loadtxt(densityFiles.format(i), delimiter=",")
            for i in range(numExistingFiles)
        ]
    )

    return overallParams, overallSimResults, overallDensityEvals


def calcWalkerAcceptance(
    model: Model,
    slice: np.ndarray,
    numWalkers: int,
    numBurnSamples: int,
    result_manager: ResultManager,
):
    """Calculate the acceptance ratio for each individual walker of the emcee chain.
        This is especially important to find "zombie" walkers, that are never moving.

    Input: model
           numBurnSamples (integer number of ignored first samples of each chain)
           numWalkers (integer number of emcee walkers) that were used for the emcee chain which is analyzed here
           result_manager (ResultManager object) that manages the data from the emcee chain which is analyzed here

    Output: acceptanceRatios (np.array of size numWalkers)

    Args:
      model: Model:
      slice: np.ndarray:
      numWalkers: int:
      numBurnSamples: int:
      result_manager: ResultManager:

    Returns:

    """

    # load the emcee parameter chain
    params = np.loadtxt(
        result_manager.getSlicePath(slice) + "/OverallParams.csv",
        delimiter=",",
        ndmin=2,
    )[numBurnSamples:, :]

    # calculate the number of steps each walker walked
    # subtract 1 because we count the steps between the parameters
    numSteps = int(params.shape[0] / numWalkers) - 1

    # Unflatten the parameter chain and count the number of accepted steps for each walker
    params = params.reshape(numSteps + 1, numWalkers, model.paramDim)

    # Build a boolean array that is true if the parameters of the current step are the same as the parameters of the next step and sum over it
    # If the parameters are the same, the step is not accepted and we add 0 to the number of accepted steps
    # If the parameters are different, the step is accepted and we add 1 to the number of accepted steps
    numAcceptedSteps = np.sum(
        np.any(params[1:, :, :] != params[:-1, :, :], axis=2),
        axis=0,
    )

    # calculate the acceptance ratio by dividing the number of accepted steps by the overall number of steps
    acceptanceRatios = numAcceptedSteps / numSteps

    return acceptanceRatios


# TODO: Benjamin mach mal!
# Dependeny matrix definieren
# Cis beliebig aber marginals nicht 0 -> central param als start wählen
# Ergebnisse müssen für verschiedene Marginals gespeichert werden z.b. Q1, Q2

# Defininiere param abhängig in blocken durch slices
# E.g.: s1 = [0,1,2], s2 = [3], s4 = [4,5]
# -> slices = [s1,s2,s3] oder als np array. einzelne slices müssen np. arrays sein für np indexierung
# Alles was an emcee geht muss reduzierte dimension haben
# Walker init, paramdim, ?
# Und am wichtigsten: evalLogTransformedDensity braucht slice (+default value) parameter
# evalLogTransformedDensity bekommt evtl. reduzierte param "version" und wir müssen vollen vektor zusammen aus centralParam
# rekonstruieren

# Inference takes slices
# parameter sampling and sampling takes one slice
# other functions have to be also adapted because of all the file names

# TODO generell: inference: gitter vs samplingbasiert: nutzer auswählen lassen
