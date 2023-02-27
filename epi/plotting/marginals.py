import numpy as np

from epi.core.kde import calcKernelWidth, evalKDEGauss
from epi.core.model import Model


def calcDataMarginals(model: Model, resolution: int) -> None:
    """Evaluate the one-dimensional marginals of the original data over equi-distant grids.
        The stored evaluations can then be used for result visualization.

    :parameter model: The model which manages the data and provides the visualization grid
    :parameter resolution: defines the number of grid points for each marginal evaluation is directly proportional to the runtime
    :return: None, but stores results as files
    """

    # Load data, data standard deviations and model characteristics for the specified model.
    (
        dataDim,
        data,
        dataStdevs,
    ) = model.dataLoader()

    # Create containers for the data marginal evaluations.
    trueDataMarginals = np.zeros((resolution, dataDim))

    # Load the grid over which the data marginal will be evaluated
    dataGrid, _ = model.generateVisualizationGrid(resolution)

    # Loop over each simulation result dimension and marginalize over the rest.
    for dim in range(dataDim):
        # The 1D-arrays of true data have to be casted to 2D-arrays, as this format is obligatory for kernel density estimation.
        marginalData = np.zeros((data.shape[0], 1))
        marginalData[:, 0] = data[:, dim]

        # Loop over all grid points and evaluate the 1D kernel marginal density estimation of the data sample.
        for i in range(resolution):
            trueDataMarginals[i, dim] = evalKDEGauss(
                marginalData,
                np.array([dataGrid[i, dim]]),
                np.array([dataStdevs[dim]]),
            )

    # Store the marginal KDE approximation of the data
    np.savetxt(
        model.getApplicationPath() + "/Plots/trueDataMarginals.csv",
        trueDataMarginals,
        delimiter=",",
    )


def calcEmceeSimResultsMarginals(
    model: Model, numBurnSamples: int, occurrence: int, resolution: int
) -> None:
    """Evaluate the one-dimensional marginals of the emcee sampling simulation results over equi-distant grids.
        The stores evaluations can then be used for result visualization.

    :param model:
    :param numBurnSamples: (Number of ignored first samples of each chain), defaults to 20% of all samples
    :param occurence: (step of sampling from chains), defaults to numWalkers+1 (ensures that the chosen samples are nearly uncorrelated)
    :param resolution: (defines the number of grid points for each marginal evaluation is directly proportional to the runtime), defaults to 100
    :return: None, except for stored files
    """

    # Load the emcee simulation results chain
    simResults = np.loadtxt(
        model.getApplicationPath() + "/OverallSimResults.csv",
        delimiter=",",
        ndmin=2,
    )[numBurnSamples::occurrence, :]

    # Load data, data standard deviations and model characteristics for the specified model.
    (
        dataDim,
        data,
        dataStdevs,
    ) = model.dataLoader()

    # Create containers for the simulation results marginal evaluations.
    inferredDataMarginals = np.zeros((resolution, dataDim))

    # Load the grid over which the simulation results marginal will be evaluated
    dataGrid, _ = model.generateVisualizationGrid(resolution)

    # Loop over each data dimension and marginalize over the rest.
    for dim in range(dataDim):
        # The 1D-arrays of simulation resultshave to be casted to 2D-arrays, as this format is obligatory for kernel density estimation.
        marginalSimResults = np.zeros((simResults.shape[0], 1))
        marginalSimResults[:, 0] = simResults[:, dim]

        # Loop over all grid points and evaluate the 1D kernel marginal density estimation of the emcee simulation results.
        for i in range(resolution):
            inferredDataMarginals[i, dim] = evalKDEGauss(
                marginalSimResults,
                np.array([dataGrid[i, dim]]),
                np.array([dataStdevs[dim]]),
            )

    # Store the marginal KDE approximation of the simulation results emcee sample
    np.savetxt(
        model.getApplicationPath() + "/Plots/inferredDataMarginals.csv",
        inferredDataMarginals,
        delimiter=",",
    )


def calcParamMarginals(
    model: Model, numBurnSamples: int, occurrence: int, resolution: int
) -> None:
    """Evaluate the one-dimensional marginals of the emcee sampling parameters (and potentially true parameters) over equi-distant grids.
        The stores evaluations can then be used for result visualization.

    :param model:(model ID)
    :param numBurnSamples: (Number of ignored first samples of each chain), defaults to 20% of all samples
    :param occurrence: (step of sampling from chains), defaults to numWalkers+1 (ensures that the chosen samples are nearly uncorrelated)
    :param resolution: (defines the number of grid points for each marginal evaluation is directly proportional to the runtime), defaults to 100
    :return: None, except for stored files
    """

    # If the model name indicates an artificial setting, indicate that true parameter information is available
    artificialModel = model.isArtificial()

    # Load the emcee parameter chain
    paramChain = np.loadtxt(
        model.getApplicationPath() + "/OverallParams.csv",
        delimiter=",",
        ndmin=2,
    )[numBurnSamples::occurrence, :]

    paramDim = model.paramDim

    # Define the standard deviation for plotting the parameters based on the sampled parameters and not the true ones.
    paramStdevs = calcKernelWidth(paramChain)

    # Create containers for the parameter marginal evaluations and the underlying grid.
    _, paramGrid = model.generateVisualizationGrid(resolution)
    inferredParamMarginals = np.zeros((resolution, paramDim))

    # If there are true parameter values available, load them and allocate storage similar to the just-defined one.
    if artificialModel:
        trueParamSample, _ = model.paramLoader()
        trueParamMarginals = np.zeros((resolution, paramDim))

    # Loop over each parameter dimension and marginalize over the rest.
    for dim in range(paramDim):
        # As the kernel density estimators only work for 2D-arrays of data, we have to cast the column of parameter samples into a 1-column matrix (or 2D-array).
        marginalParamChain = np.zeros((paramChain.shape[0], 1))
        marginalParamChain[:, 0] = paramChain[:, dim]

        # If there is true parameter information available, we have to do the same type cast for the true parameter samples.
        if artificialModel:
            trueMarginalParamSample = np.zeros((trueParamSample.shape[0], 1))
            trueMarginalParamSample[:, 0] = trueParamSample[:, dim]

        # Loop over all grid points and evaluate the 1D kernel density estimation of the reconstructed marginal parameter distribution.
        for i in range(resolution):
            inferredParamMarginals[i, dim] = evalKDEGauss(
                marginalParamChain,
                np.array([paramGrid[i, dim]]),
                np.array(paramStdevs[dim]),
            )

            # If true parameter information is available, evaluate a similar 1D marginal distribution based on the true parameter samples.
            if artificialModel:
                trueParamMarginals[i, dim] = evalKDEGauss(
                    trueMarginalParamSample,
                    np.array([paramGrid[i, dim]]),
                    np.array(paramStdevs[dim]),
                )

    # Store the (potentially 2) marginal distribution(s) for later plotting
    np.savetxt(
        model.getApplicationPath() + "/Plots/inferredParamMarginals.csv",
        inferredParamMarginals,
        delimiter=",",
    )

    if artificialModel:
        np.savetxt(
            "Applications/"
            + model.name
            + "/Plots/trueParamMarginals.csv",
            trueParamMarginals,
            delimiter=",",
        )
