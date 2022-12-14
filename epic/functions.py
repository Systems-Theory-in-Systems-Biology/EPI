import numpy as np

from epic.kernel_density_estimation import evalKDEGauss, calcKernelWidth
from epic.models.model import ArtificialModelInterface, Model

def evalLogTransformedDensity(param, model:Model, data, dataStdevs):
    """Given a simulation model, its derivative and corresponding data, evaluate the natural log of the parameter density that is the backtransformed data distribution.
        This function is intended to be used with the emcee sampler and can be implemented more efficiently at some points.

    Input: param (parameter for which the transformed density shall be evaluated)
           modelName (model ID)
           data (data for the model: 2D array with shape (#numDataPoints, #dataDim))
           dataStdevs (array of suitable kernel standard deviations for each data dimension)
    Output: logTransformedDensity (natural log of parameter density at the point param)
          : allRes (array concatenation of parameters, simulation results and evaluated density, stored as "blob" by the emcee sampler)
    """

    # Define model-specific lower...
    paramsLowerLimitsDict = {
        "Linear": np.array([-10.0, -10.0]),
        "LinearODE": np.array([-10.0, -10.0]),
        "Temperature": np.array([0]),
        "TemperatureArtificial": np.array([0]),
        "Corona": np.array([-4.5, -2.0, -2.0]),
        "CoronaArtificial": np.array([-2.5, -0.75, 0.0]),
        "Stock": np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0]),
        "StockArtificial": np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
    }

    # ... and upper borders for sampling to avoid parameter regions where the simulation can only be evaluated instably.
    paramsUpperLimitsDict = {
        "Linear": np.array([11.0, 11.0]),
        "LinearODE": np.array([23.0, 23.0]),
        "Temperature": np.array([np.pi / 2]),
        "TemperatureArtificial": np.array([np.pi / 2]),
        "Corona": np.array([0.5, 3.0, 3.0]),
        "CoronaArtificial": np.array([-1.0, 0.75, 1.5]),
        "Stock": np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        "StockArtificial": np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
    }

    # Check if the tried parameter is within the just-defined bounds and return the lowest possible log density if not.
    if (np.any(param < paramsLowerLimitsDict[model.getModelName()])) or (
        np.any(param > paramsUpperLimitsDict[model.getModelName()])
    ):
        print("parameters outside of predefines range")
        return -np.inf, np.zeros(param.shape[0] + data.shape[1] + 1)

    # If the parameter is within the valid ranges...
    else:
        # Evaluate the simulation result for the specified parameter.
        simRes = model(param)

        # Evaluate the data density in the simulation result.
        densityEvaluation = evalKDEGauss(data, simRes, dataStdevs)

        # Calculate the simulation model's pseudo-determinant in the parameter point (also called the correction factor).
        correction = model.correction(param)

        # Multiply data density and correction factor.
        trafoDensityEvaluation = densityEvaluation * correction

        # Use the log of the transformed density because emcee requires this.
        logTransformedDensity = np.log(trafoDensityEvaluation)

        # Store the current parameter, its simulation result as well as its density in a large vector that is stored separately by emcee.
        allRes = np.concatenate(
            (param, simRes, np.array([trafoDensityEvaluation]))
        )

        return logTransformedDensity, allRes


# TODO: defaults to [DefaultParamVal], resolution=100?
# This should bei either set here and in the other functions or not mentioned at all
def calcDataMarginals(model:Model, resolution):
    """Evaluate the one-dimensional marginals of the original data over equi-distant grids.
        The stored evaluations can then be used for result visualization.

    :parameter modelName: model ID
    :parameter resolution: defines the number of grid points for each marginal evaluation is directly proportional to the runtime
    :return: None, but stores results as files
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
        "Applications/" + model.getModelName() + "/Plots/trueDataMarginals.csv",
        trueDataMarginals,
        delimiter=",",
    )


def calcEmceeSimResultsMarginals(
    model:Model, numBurnSamples, occurrence, resolution
):
    """Evaluate the one-dimensional marginals of the emcee sampling simulation results over equi-distant grids.
        The stores evaluations can then be used for result visualization.

    Input: modelName (model ID)
           numBurnSamples (Number of ignored first samples of each chain)
           occurence (step of sampling from chains)
           resolution (defines the number of grid points for each marginal evaluation is directly proportional to the runtime)
    Output: <none except for stored files>

    Standard parameters : numBurnSamples = 20% of all samples
                          occurence = numWalkers+1 (ensures that the chosen samples are nearly uncorrelated)
                          resolution = 100
    """

    # Load the emcee simulation results chain
    simResults = np.loadtxt(
        "Applications/" + model.getModelName() + "/OverallSimResults.csv", delimiter=","
    )[numBurnSamples::occurrence, :]

    # Load data, data standard deviations and model characteristics for the specified model.
    (
        paramDim,
        dataDim,
        numDataPoints,
        centralParam,
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
        "Applications/" + model.getModelName() + "/Plots/inferredDataMarginals.csv",
        inferredDataMarginals,
        delimiter=",",
    )


def calcParamMarginals(model:Model, numBurnSamples, occurrence, resolution):
    """Evaluate the one-dimensional marginals of the emcee sampling parameters (and potentially true parameters) over equi-distant grids.
        The stores evaluations can then be used for result visualization.

    Input: modelName (model ID)
           numBurnSamples (Number of ignored first samples of each chain)
           occurence (step of sampling from chains)
           resolution (defines the number of grid points for each marginal evaluation is directly proportional to the runtime)
    Output: <none except for stored files>

    Standard parameters : numBurnSamples = 20% of all samples
                          occurence = numWalkers+1 (ensures that the chosen samples are nearly uncorrelated)
                          resolution = 100
    """

    # If the model name indicates an artificial setting, indicate that true parameter information is available
    artificialBool = issubclass(model.__class__, ArtificialModelInterface)

    # Load the emcee parameter chain
    paramChain = np.loadtxt(
        "Applications/" + model.getModelName() + "/OverallParams.csv", delimiter=","
    )[numBurnSamples::occurrence, :]

    # Load data, data standard deviations and model characteristics for the specified model.
    (
        paramDim,
        dataDim,
        numDataPoints,
        centralParam,
        data,
        dataStdevs,
    ) = model.dataLoader()

    # Define the standard deviation for plotting the parameters based on the sampled parameters and not the true ones.
    paramStdevs = calcKernelWidth(paramChain)

    # Create containers for the parameter marginal evaluations and the underlying grid.
    _, paramGrid = model.generateVisualizationGrid(resolution)
    inferredParamMarginals = np.zeros((resolution, paramDim))

    # If there are true parameter values available, load them and allocate storage similar to the just-defined one.
    if artificialBool == 1:
        trueParamSample, _ = model.paramLoader()
        trueParamMarginals = np.zeros((resolution, paramDim))

    # Loop over each parameter dimension and marginalize over the rest.
    for dim in range(paramDim):
        # As the kernel density estimators only work for 2D-arrays of data, we have to cast the column of parameter samples into a 1-column matrix (or 2D-array).
        marginalParamChain = np.zeros((paramChain.shape[0], 1))
        marginalParamChain[:, 0] = paramChain[:, dim]

        # If there is true parameter information available, we have to do the same type cast for the true parameter samples.
        if artificialBool == 1:
            trueMarginalParamSample = np.zeros((trueParamSample.shape[0], 1))
            trueMarginalParamSample[:, 0] = trueParamSample[:, dim]

        # Loop over all grid points and evaluate the 1D kernel density estimation of the reconstructed marginal parameter distribution.
        for i in range(resolution):
            inferredParamMarginals[i, dim] = evalKDEGauss(
                marginalParamChain,
                np.array([paramGrid[i, dim]]),
                np.array(paramStdevs[dim]),
            )

            # If true parameter information is available, evaluate a similat 1D marginal distribution based on the true parameter samples.
            if artificialBool == 1:
                trueParamMarginals[i, dim] = evalKDEGauss(
                    trueMarginalParamSample,
                    np.array([paramGrid[i, dim]]),
                    np.array(paramStdevs[dim]),
                )

    # Store the (potentially 2) marginal distribution(s) for later plotting
    np.savetxt(
        "Applications/" + model.getModelName() + "/Plots/inferredParamMarginals.csv",
        inferredParamMarginals,
        delimiter=",",
    )

    if artificialBool == 1:
        np.savetxt(
            "Applications/" + model.getModelName() + "/Plots/trueParamMarginals.csv",
            trueParamMarginals,
            delimiter=",",
        )
