import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from epic.functions import evalKDECauchy, evalKDEGauss, evalLogTransformedDensity
from epic.models import calcStdevs, dataLoader, modelLoader, paramLoader

colorQ = np.array([255.0, 147.0, 79.0]) / 255.0
colorQApprox = np.array([204.0, 45.0, 53.0]) / 255.0
colorY = np.array([5.0, 142.0, 217.0]) / 255.0
colorYApprox = np.array([132.0, 143.0, 162.0]) / 255.0

colorExtra1 = np.array([45.0, 49.0, 66.0]) / 255.0
colorExtra2 = np.array([255.0, 218.0, 174.0]) / 255.0

# Plot the 1D kernel density estimation of different data sets over a grid

# Input: data (array of array of samples)
#        stdevs (one kernel standard deviation for each dimension)
# Output: None (except for figures)


def plotKDEoverGrid(data, stdevs):
    resolution = 51

    for dim in range(data.shape[1]):
        minData = np.amin(data[:, dim])
        maxData = np.amax(data[:, dim])
        diffData = maxData - minData

        # evalPoints = np.linspace(minData-0.25*diffData, maxData+0.25*diffData, resolution)
        evalPoints = np.linspace(-5.0, 5.0, resolution)
        evaluations = np.zeros(resolution)

        for i in range(resolution):
            evaluations[i] = evalKDEGauss(
                np.transpose(np.array([data[:, dim]])),
                np.array([evalPoints[i]]),
                np.array([stdevs[dim]]),
            )

        plt.figure()
        plt.plot(evalPoints, evaluations)
        plt.hist(data[:, dim], bins=evalPoints, density=True)
        plt.show()


# The temperature model for artificial and real data is evaluated over a grid

# Input: modelName (model ID)
# Output: None (except for figures)


def plotGridResults(modelName):
    model, modelJac = modelLoader(modelName)
    (
        paramDim,
        dataDim,
        numDataPoints,
        centralParam,
        data,
        stdevs,
    ) = dataLoader(modelName)

    plt.rcParams.update({"font.size": 13})

    # TODO: Fix var names etc.
    trueParams = paramStdevs = None
    if "Artificial" in modelName:
        trueParams, paramStdevs = paramLoader(modelName)

    if modelName == "TemperatureArtificial":
        rawTrueLatitudes = np.loadtxt(
            "Applications/Temperature/Latitudes.csv", delimiter=","
        )
        trueLatitudes = np.zeros((rawTrueLatitudes.shape[0], 1))
        trueLatitudes[:, 0] = rawTrueLatitudes

        trueLatitudesStdevs = calcStdevs(trueLatitudes, modelName)

        resolution = 1000

        latitudesGrid = np.linspace(0, np.pi / 2.0, resolution)
        temperaturesGrid = np.linspace(-5, 30.0, resolution)

        trueDensity = np.zeros(resolution)
        trafoDensity = np.zeros(resolution)
        simulatedTemperatures = np.zeros(resolution)

        for i in range(resolution):
            trueDensity[i] = evalKDEGauss(
                trueLatitudes,
                np.array([latitudesGrid[i]]),
                trueLatitudesStdevs,
            )
            trafoDensity[i], _ = evalLogTransformedDensity(
                model, modelJac, np.array([latitudesGrid[i]]), data, stdevs
            )
            simulatedTemperatures[i] = evalKDEGauss(
                data, np.array([temperaturesGrid[i]]), stdevs
            )

        # plot true latitudes sample
        plt.figure(figsize=(6, 1))
        plt.axis([-0.05, np.pi / 2.0 + 0.05, -0.02, 0.04])
        plt.xlabel(r"Latitude $q_i$")
        plt.yticks([])
        plt.scatter(
            trueLatitudes[:, 0],
            np.zeros(trueLatitudes.shape[0]),
            marker=r"d",
            color=colorQ,
            alpha=0.1,
            label="True Latitudes (Sample)",
        )
        plt.legend()
        plt.savefig(
            "Images/TemperatureArtificial/TrueLatSample.svg",
            bbox_inches="tight",
        )
        plt.show()

        # plot true latitudes KDE
        plt.figure(figsize=(6, 4))
        plt.axis([-0.05, np.pi / 2.0 + 0.05, 0.0, 1.5])
        plt.xlabel(r"Latitude $q_i$")
        plt.ylabel(r"Density $\phi_Q(q_i)$")
        plt.plot(
            latitudesGrid,
            trueDensity,
            linewidth=3.0,
            color=colorQ,
            label="True Latitudes (KDE)",
        )
        plt.legend()
        plt.savefig(
            "Images/TemperatureArtificial/TrueLatKDE.svg", bbox_inches="tight"
        )
        plt.show()

        # plot simulated temperature sample
        plt.figure(figsize=(6, 1))
        plt.axis([-6.2, 31.2, -0.02, 0.04])
        plt.xlabel(r"Temperature $y_i$")
        plt.yticks([])
        plt.scatter(
            data,
            np.zeros(data.shape[0]),
            marker=r"d",
            color=colorY,
            alpha=0.1,
            label="Simulated Temperatures (Sample)",
        )
        plt.legend()
        plt.savefig(
            "Images/TemperatureArtificial/SimTempSample.svg",
            bbox_inches="tight",
        )
        plt.show()

        # plot simulated temperatures KDE
        plt.figure(figsize=(6, 4))
        plt.axis([-6.2, 31.2, 0.0, 0.11])
        plt.xlabel(r"Temperature $y_i$")
        plt.ylabel(r"Density $\phi_Y(y_i)$")
        plt.plot(
            temperaturesGrid,
            simulatedTemperatures,
            linewidth=3.0,
            color=colorY,
            label="Simulated Temperatures (KDE)",
        )
        plt.legend()
        plt.savefig(
            "Images/TemperatureArtificial/SimTempKDE.svg", bbox_inches="tight"
        )
        plt.show()

        # plot ITA inferred latitude density
        plt.figure(figsize=(6, 4))
        plt.axis([-0.05, np.pi / 2.0 + 0.05, 0.0, 1.5])
        plt.xlabel(r"Latitude $q_i$")
        plt.ylabel(r"Density $\phi_Q(q_i)$")
        plt.plot(
            latitudesGrid,
            trueDensity,
            linewidth=3.0,
            color=colorQ,
            label="True Latitudes (KDE)",
        )
        plt.plot(
            latitudesGrid,
            trafoDensity,
            linewidth=3.0,
            color=colorQApprox,
            label="Inferred Latitudes",
        )
        plt.legend()
        plt.savefig(
            "Images/TemperatureArtificial/TrueLatVsITAKDE.svg",
            bbox_inches="tight",
        )
        plt.show()

        # plot temperature art style
        plt.figure(figsize=(6, 4))
        plt.axis(
            [np.amin(data[:, 0]) - 1, np.amax(data[:, 0]) + 1, -0.02, 0.1]
        )
        plt.axis("off")

        for i in range(temperaturesGrid.shape[0] - 1):
            relativeIndex = i / (temperaturesGrid.shape[0] - 1)
            cwcolor = [relativeIndex, 0.0, 1.0 - relativeIndex]
            plt.plot(
                temperaturesGrid[i : i + 2],
                simulatedTemperatures[i : i + 2],
                linewidth=3.0,
                color=cwcolor,
            )

        minData = np.amin(data[:, 0])
        maxData = np.amax(data[:, 0])

        # for j in range(data.shape[0]):
        #    cwcolor = [(data[j,0]-minData)/(maxData-minData),0.0,1-(data[j,0]-minData)/(maxData-minData)]
        #    plt.scatter(data[j], 0.0, marker = r"d", color = cwcolor, alpha = 0.1)

        plt.savefig(
            "Images/TemperatureArtificial/ArtTemp.svg", bbox_inches="tight"
        )
        plt.show()

        # plot latitudes art style
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.axis([-2.25, 0.0, 0.0, 2.25])

        for i in range(latitudesGrid.shape[0] - 1):
            relativeIndex = i / (latitudesGrid.shape[0] - 1)
            cwcolor = [
                relativeIndex,
                (112 + 144 * relativeIndex) / 256.0,
                (192 + 64 * relativeIndex) / 256.0,
            ]

            plt.plot(
                -np.cos(latitudesGrid[i : i + 2])
                * (1 + trafoDensity[i : i + 2]),
                np.sin(latitudesGrid[i : i + 2])
                * (1 + trafoDensity[i : i + 2]),
                linewidth=3.0,
                color=cwcolor,
            )

            plt.plot(
                -np.cos(latitudesGrid[i : i + 2]),
                np.sin(latitudesGrid[i : i + 2]),
                alpha=0.05,
                linewidth=3.0,
                color=cwcolor,
            )

        plt.savefig(
            "Images/TemperatureArtificial/ArtLat.svg", bbox_inches="tight"
        )
        plt.show()

    elif modelName == "Temperature":
        rawTrueLatitudes = np.loadtxt(
            "Applications/Temperature/Latitudes.csv", delimiter=","
        )
        trueLatitudes = np.zeros((rawTrueLatitudes.shape[0], 1))
        trueLatitudes[:, 0] = rawTrueLatitudes

        trueLatitudesStdevs = calcStdevs(trueLatitudes, modelName)

        resolution = 400

        latitudesGrid = np.linspace(0, np.pi / 2.0, resolution)
        temperaturesGrid = np.linspace(-5, 30.0, resolution)

        trueDensity = np.zeros(resolution)
        trafoDensity = np.zeros(resolution)
        measuredTemperatures = np.zeros(resolution)

        for i in range(resolution):
            trueDensity[i] = evalKDEGauss(
                trueLatitudes,
                np.array([latitudesGrid[i]]),
                trueLatitudesStdevs,
            )
            trafoDensity[i], _ = evalLogTransformedDensity(
                model, modelJac, np.array([latitudesGrid[i]]), data, stdevs
            )
            measuredTemperatures[i] = evalKDEGauss(
                data, np.array([temperaturesGrid[i]]), stdevs
            )

        sampleSize = 1000
        trafoDensitySample = np.random.choice(
            latitudesGrid,
            sampleSize,
            replace="True",
            p=trafoDensity / np.sum(trafoDensity),
        )

        inferredTemperaturesSample = np.zeros((sampleSize, 1))
        for i in range(sampleSize):
            inferredTemperaturesSample[i, 0] = model(
                np.array([trafoDensitySample[i]])
            )

        inferredTemperatures = np.zeros(resolution)
        for i in range(resolution):
            inferredTemperatures[i] = evalKDEGauss(
                inferredTemperaturesSample,
                np.array([temperaturesGrid[i]]),
                stdevs,
            )

        # plot true latitudes sample
        plt.figure(figsize=(6, 1))
        plt.axis([-0.05, np.pi / 2.0 + 0.05, -0.02, 0.04])
        plt.xlabel(r"Latitude $q_i$")
        plt.yticks([])
        plt.scatter(
            trueLatitudes[:, 0],
            np.zeros(trueLatitudes.shape[0]),
            marker=r"d",
            color=colorQ,
            alpha=0.1,
            label="True Latitudes (Sample)",
        )
        plt.legend()
        plt.savefig(
            "Images/Temperature/TrueLatSample.svg", bbox_inches="tight"
        )
        plt.show()

        # plot true latitudes KDE
        plt.figure(figsize=(6, 4))
        plt.axis([-0.05, np.pi / 2.0 + 0.05, 0.0, 1.6])
        plt.xlabel(r"Latitude $q_i$")
        plt.ylabel(r"Density $\phi_Q(q_i)$")
        plt.plot(
            latitudesGrid,
            trueDensity,
            linewidth=3.0,
            color=colorQ,
            label="True Latitudes (KDE)",
        )
        plt.legend()
        plt.savefig("Images/Temperature/TrueLatKDE.svg", bbox_inches="tight")
        plt.show()

        # plot simulated temperature sample
        plt.figure(figsize=(6, 1))
        plt.axis([-6.2, 31.2, -0.02, 0.04])
        plt.xlabel(r"Temperature $y_i$")
        plt.yticks([])
        plt.scatter(
            data,
            np.zeros(data.shape[0]),
            marker=r"d",
            color=colorY,
            alpha=0.1,
            label="Measured Temperatures (Sample)",
        )
        plt.legend()
        plt.savefig(
            "Images/Temperature/MeasTempSample.svg", bbox_inches="tight"
        )
        plt.show()

        # plot simulated temperatures KDE
        plt.figure(figsize=(6, 4))
        plt.axis([-6.2, 31.2, 0.0, 0.08])
        plt.xlabel(r"Temperature $y_i$")
        plt.ylabel(r"Density $\phi_Y(y_i)$")
        plt.plot(
            temperaturesGrid,
            measuredTemperatures,
            linewidth=3.0,
            color=colorY,
            label="Measured Temperatures (KDE)",
        )
        plt.legend()
        plt.savefig("Images/Temperature/MeasTempKDE.svg", bbox_inches="tight")
        plt.show()

        # plot ITA inferred latitude density in comparison to true latitudes
        plt.figure(figsize=(6, 4))
        plt.axis([-0.05, np.pi / 2.0 + 0.05, 0.0, 1.6])
        plt.xlabel(r"Latitude $q_i$")
        plt.ylabel(r"Density $\phi_Q(q_i)$")
        plt.plot(
            latitudesGrid,
            trueDensity,
            linewidth=3.0,
            color=colorQ,
            label="True Latitudes (KDE)",
        )
        plt.plot(
            latitudesGrid,
            trafoDensity,
            linewidth=3.0,
            color=colorQApprox,
            label="Inferred Latitudes",
        )
        plt.legend()
        plt.savefig(
            "Images/Temperature/TrueLatVsITAKDE.svg", bbox_inches="tight"
        )
        plt.show()

        # plot ITA inferred latitude density alone
        plt.figure(figsize=(6, 4))
        plt.axis([-0.05, np.pi / 2.0 + 0.05, 0.0, 1.6])
        plt.xlabel(r"Latitude $q_i$")
        plt.ylabel(r"Density $\phi_Q(q_i)$")
        plt.plot(
            latitudesGrid,
            trafoDensity,
            linewidth=3.0,
            color=colorQApprox,
            label="Inferred Latitudes",
        )
        plt.legend()
        plt.savefig("Images/Temperature/ITAKDE.svg", bbox_inches="tight")
        plt.show()

        # plot inferred latitudes sample
        plt.figure(figsize=(6, 1))
        plt.axis([-0.05, np.pi / 2.0 + 0.05, -0.02, 0.04])
        plt.xlabel(r"Latitude $q_i$")
        plt.yticks([])
        plt.scatter(
            trafoDensitySample,
            np.zeros(sampleSize),
            marker=r"d",
            color=colorQApprox,
            alpha=0.1,
            label="Inferred Latitudes (Sample)",
        )
        plt.legend()
        plt.savefig(
            "Images/Temperature/InferredLatSample.svg", bbox_inches="tight"
        )
        plt.show()

        # plot inferred temperatures sample
        plt.figure(figsize=(6, 2))
        plt.axis([-6.2, 31.2, -0.05, 0.04])
        plt.xlabel(r"Temperature $y_i$")
        plt.yticks([])
        plt.scatter(
            data,
            np.zeros(data.shape[0]),
            marker=r"d",
            color=colorY,
            alpha=0.1,
            label="Measured Temperatures (Sample)",
        )
        plt.scatter(
            inferredTemperaturesSample,
            0.02 * np.ones(sampleSize),
            marker=r"d",
            color=colorYApprox,
            alpha=0.1,
            label="Inferred Temperatures (Sample)",
        )
        plt.legend()
        plt.savefig(
            "Images/Temperature/InferredTempSample.svg", bbox_inches="tight"
        )
        plt.show()

        # plot inferred vs measured temperatures KDE
        plt.figure(figsize=(6, 4))
        plt.axis([-6.2, 31.2, 0.0, 0.08])
        plt.xlabel(r"Temperature $y_i$")
        plt.ylabel(r"Density $\phi_Y(y_i)$")
        plt.plot(
            temperaturesGrid,
            measuredTemperatures,
            linewidth=3.0,
            color=colorY,
            label="Measured Temperatures (KDE)",
        )
        plt.plot(
            temperaturesGrid,
            inferredTemperatures,
            linewidth=3.0,
            color=colorYApprox,
            label="Inferred Temperatures (KDE)",
        )
        plt.legend()
        plt.savefig(
            "Images/Temperature/MeasTempVsITAKDE.svg", bbox_inches="tight"
        )
        plt.show()

    else:
        print("Function only defined for other scenarios")


# plot sampling results in comparison to true results

# Input: modelName (model ID)
#        numBurnSamples (Ignore the first samples of each chain)
#        occurence (step of sampling from chains)
#        artificialBool (indicated whether true parameter information is available and shall be loaded)
# Output: None (except for figures)


def plotEmceeResults(modelName, numBurnSamples, occurrence, artificialBool):
    resolution = 100

    (
        paramDim,
        dataDim,
        numDataPoints,
        centralParam,
        data,
        dataStdevs,
    ) = dataLoader(modelName)

    densityEvals = np.loadtxt(
        "Applications/" + modelName + "/OverallDensityEvals.csv", delimiter=","
    )[numBurnSamples::occurrence]
    simResults = np.loadtxt(
        "Applications/" + modelName + "/OverallSimResults.csv", delimiter=","
    )[numBurnSamples::occurrence, :]
    paramChain = np.loadtxt(
        "Applications/" + modelName + "/OverallParams.csv", delimiter=","
    )[numBurnSamples::occurrence, :]

    if artificialBool == 1:
        trueParams, paramStdevs = paramLoader(modelName)
    else:
        paramStdevs = calcStdevs(paramChain)

    print("Successfully loaded data!")

    # plot sampled parameters in comparison to true ones
    for dim in range(paramDim):
        evaluations = np.zeros(resolution)
        singleParamChain = np.zeros((paramChain.shape[0], 1))
        singleParamChain[:, 0] = paramChain[:, dim]

        paramGrid = np.linspace(
            np.amin(paramChain[:, dim]),
            np.amax(paramChain[:, dim]),
            resolution,
        )

        if artificialBool == 1:
            trueEvaluations = np.zeros(resolution)
            trueSingleParamSample = np.zeros((trueParams.shape[0], 1))
            trueSingleParamSample[:, 0] = trueParams[:, dim]

        for i in range(resolution):
            evaluations[i] = evalKDEGauss(
                singleParamChain,
                np.array([paramGrid[i]]),
                np.array(paramStdevs[dim]),
            )

            if artificialBool == 1:
                trueEvaluations[i] = evalKDEGauss(
                    trueSingleParamSample,
                    np.array([paramGrid[i]]),
                    np.array(paramStdevs[dim]),
                )

        plt.figure()
        plt.plot(
            paramGrid,
            evaluations,
            c=colorQApprox,
            label="parameter estimation",
        )

        if artificialBool == 1:
            plt.plot(
                paramGrid, trueEvaluations, c=colorQ, label="parameter truth"
            )

        plt.legend()
        plt.plot()

    # plot sampled simulation results in comparison to original data
    for dim in range(dataDim):
        simResEvaluations = np.zeros(resolution)
        dataEvaluations = np.zeros(resolution)

        singleSimResults = np.zeros((simResults.shape[0], 1))
        singleSimResults[:, 0] = simResults[:, dim]
        singleData = np.zeros((data.shape[0], 1))
        singleData[:, 0] = data[:, dim]

        globalMin = min(
            np.amin(singleData[:, 0]), np.amin(singleSimResults[:, 0])
        )
        globalMax = max(
            np.amax(singleData[:, 0]), np.amax(singleSimResults[:, 0])
        )

        evalGrid = np.linspace(globalMin, globalMax, resolution)

        for i in range(resolution):
            simResEvaluations[i] = evalKDEGauss(
                singleSimResults,
                np.array([evalGrid[i]]),
                np.array([dataStdevs[dim]]),
            )
            dataEvaluations[i] = evalKDEGauss(
                singleData,
                np.array([evalGrid[i]]),
                np.array([dataStdevs[dim]]),
            )

        plt.figure()
        plt.plot(
            evalGrid,
            simResEvaluations,
            c=colorYApprox,
            label="data reconstruction",
        )
        plt.plot(evalGrid, dataEvaluations, c=colorY, label="data truth")
        plt.legend()
        plt.show()

    return 0


def plotDataMarginals(modelName):
    (
        paramDim,
        dataDim,
        numDataPoints,
        centralParam,
        data,
        dataStdevs,
    ) = dataLoader(modelName)

    dataGrid = np.loadtxt(
        "Applications/" + modelName + "/Plots/dataGrid.csv", delimiter=","
    )
    trueDataMarginals = np.loadtxt(
        "Applications/" + modelName + "/Plots/trueDataMarginals.csv",
        delimiter=",",
    )

    for dim in range(dataDim):
        plt.figure()
        plt.plot(
            dataGrid[:, dim],
            trueDataMarginals[:, dim],
            c=colorY,
            label="true data marg. KDE",
        )
        plt.hist(
            data[:, dim],
            bins=dataGrid[:, dim],
            color=np.concatenate((colorY, np.array([0.5]))),
            label="true data marg. hist.",
            density=True,
        )
        plt.legend()
        plt.show()


def plotMarginals(modelName, numBurnSamples, occurrence):
    artificialBool = 0

    if (
        (modelName == "TemperatureArtificial")
        or (modelName == "CoronaArtificial")
        or (modelName == "StockArtificial")
    ):
        artificialBool = 1

    simResults = np.loadtxt(
        "Applications/" + modelName + "/OverallSimResults.csv", delimiter=","
    )[numBurnSamples::occurrence, :]
    paramChain = np.loadtxt(
        "Applications/" + modelName + "/OverallParams.csv", delimiter=","
    )[numBurnSamples::occurrence, :]

    (
        paramDim,
        dataDim,
        numDataPoints,
        centralParam,
        data,
        dataStdevs,
    ) = dataLoader(modelName)

    paramGrid = np.loadtxt(
        "Applications/" + modelName + "/Plots/paramGrid.csv", delimiter=","
    )
    inferredParamMarginals = np.loadtxt(
        "Applications/" + modelName + "/Plots/inferredParamMarginals.csv",
        delimiter=",",
    )

    if artificialBool == 1:
        trueParams, paramStdevs = paramLoader(modelName)
        trueParamMarginals = np.loadtxt(
            "Applications/" + modelName + "/Plots/trueParamMarginals.csv",
            delimiter=",",
        )

    dataGrid = np.loadtxt(
        "Applications/" + modelName + "/Plots/dataGrid.csv", delimiter=","
    )
    trueDataMarginals = np.loadtxt(
        "Applications/" + modelName + "/Plots/trueDataMarginals.csv",
        delimiter=",",
    )
    inferredDataMarginals = np.loadtxt(
        "Applications/" + modelName + "/Plots/inferredDataMarginals.csv",
        delimiter=",",
    )

    for dim in range(paramDim):
        plt.figure()
        plt.plot(
            paramGrid[:, dim],
            inferredParamMarginals[:, dim],
            c=colorQApprox,
            label="inferred param. marg. KDE",
        )
        # plt.hist(paramChain[:,dim], bins = paramGrid[:,dim], color = np.concatenate((colorQApprox, np.array([0.5]))), label = "inferred param. marg. hist.", density = True)

        if artificialBool == 1:
            plt.plot(
                paramGrid[:, dim],
                trueParamMarginals[:, dim],
                c=colorQ,
                label="true param. marg. KDE",
            )
            # plt.hist(trueParams[:,dim], bins = paramGrid[:,dim], color = np.concatenate((colorQ, np.array([0.5]))), label = "true param. marg. hist.", density = True)

        plt.legend()
        plt.show()

    for dim in range(dataDim):
        plt.figure()
        plt.plot(
            dataGrid[:, dim],
            inferredDataMarginals[:, dim],
            c=colorYApprox,
            label="reconstr. data marg. KDE",
        )
        # plt.hist(simResults[:,dim], bins = dataGrid[:,dim], color = np.concatenate((colorYApprox, np.array([0.5]))), label = "recontr. data. marg. hist.", density = True)

        plt.plot(
            dataGrid[:, dim],
            trueDataMarginals[:, dim],
            c=colorY,
            label="true data marg. KDE",
        )
        # plt.hist(data[:,dim], bins = dataGrid[:,dim], color = np.concatenate((colorY, np.array([0.5]))), label = "true data. marg. hist.", density = True)
        plt.legend()
        plt.show()


def plotSpiderWebs(modelName, numBurnSamples, occurrence):
    """Draw each sample (row of the matrix) as a circle of linear interpolations of its dimensions.
    Loads all necessary data and subsequently calls the method singleWeb 3 or 4 times

    Input: modelName (model ID)
           numBurnSamples (Number of ignored first samples of each chain)
           occurence (step of sampling from chains)
    Output: <none>

    """

    # By default, we assume that no true parameter information is available
    artificialBool = 0

    # If the model name indicates an artificial setting, indicate that true parameter information is available
    if (
        (modelName == "TemperatureArtificial")
        or (modelName == "CoronaArtificial")
        or (modelName == "StockArtificial")
    ):
        artificialBool = 1

    # load emcee parameter samples and corresponding simulation results
    emceeParams = np.loadtxt(
        "Applications/" + modelName + "/OverallParams.csv", delimiter=","
    )[numBurnSamples::occurrence, :]
    emceeSimResults = np.loadtxt(
        "Applications/" + modelName + "/OverallSimResults.csv", delimiter=","
    )[numBurnSamples::occurrence, :]

    # load underlying data
    trueData = np.loadtxt("Data/" + modelName + "Data.csv", delimiter=",")

    # if available, load also the true parameter values
    if artificialBool == 1:
        trueParams = np.loadtxt(
            "Data/" + modelName + "Params.csv", delimiter=","
        )

    # compute the upper and lower bound of each data dimension
    # this serves as the identical scaling of every data plot
    upperBoundsSimResults = np.max(emceeSimResults, 0)
    lowerBoundsSimResults = np.min(emceeSimResults, 0)

    # do the same for the parameters
    upperBoundsParams = np.max(emceeParams, 0)
    lowerBoundsParams = np.min(emceeParams, 0)

    # set the image quality by defining dots per inch
    dpi = 1000

    # create all web figures
    emceeSimResultsWeb = singleWeb(
        emceeSimResults,
        lowerBoundsSimResults,
        upperBoundsSimResults,
        colorYApprox,
        dpi,
    )
    plt.savefig(
        "Applications/" + modelName + "/SpiderWebs/emceeSimResults.png",
        dpi=dpi,
    )

    trueDataWeb = singleWeb(
        trueData, lowerBoundsSimResults, upperBoundsSimResults, colorY, dpi
    )
    plt.savefig(
        "Applications/" + modelName + "/SpiderWebs/trueData.png", dpi=dpi
    )

    emceeParamsWeb = singleWeb(
        emceeParams, lowerBoundsParams, upperBoundsParams, colorQApprox, dpi
    )
    plt.savefig(
        "Applications/" + modelName + "/SpiderWebs/emceeParams.png", dpi=dpi
    )

    if artificialBool == 1:
        trueParamsWeb = singleWeb(
            trueParams, lowerBoundsParams, upperBoundsParams, colorQ, dpi
        )
        plt.savefig(
            "Applications/" + modelName + "/SpiderWebs/trueParams.png", dpi=dpi
        )


def singleWeb(matrix, lowerBounds, upperBounds, color, dpi):
    """Create a single spider web plot for one data or parameter matrix and given bounds

    Input: matrix (2D np.array of size #samples x #dimensions)
           lowerBounds (np.arraay of size #dimensions that defines the lower bound of all regarded values)
           upperBounds (np.arraay of size #dimensions that defines the upper bound of all regarded values)
           color (np.array with 3 entries indicating RGB values of the plot)
           dpi (int that defines the quality of the image)

    Output: web (matplotlib figure representing the single spider web)
    """

    # extract matrix dimensions
    numSamples, numDims = matrix.shape

    # create a normalized matrix where each column aka. dimension is scaled to [0, 1]
    normMatrix = (matrix - lowerBounds) / (upperBounds - lowerBounds)

    # create an augmented matrix that is identical to the normalized matrix
    # except for an additional last column that is obtained by copying the first one
    augMatrix = np.zeros((numSamples, numDims + 1))
    augMatrix[:, 0:numDims] = normMatrix

    # copy the first column to the end
    augMatrix[:, -1] = normMatrix[:, 0]

    # calculate the angles of the spiderweb equidistantly distributed between 0 and 2 pi
    angles = np.linspace(0, 2 * np.pi, numDims + 1)

    xProjFactor = np.sin(angles)
    yProjFactor = np.cos(angles)

    xProjCoords = augMatrix * xProjFactor
    yProjCoords = augMatrix * yProjFactor

    fig = plt.figure(
        figsize=(2000 / dpi, 2000 / dpi),
        dpi=dpi,
        frameon=False,
        layout="tight",
    )
    plt.axis([-1.1, 1.1, -1.1, 1.1])

    # delete the figure frame
    for pos in ["right", "top", "bottom", "left"]:
        plt.gca().spines[pos].set_visible(False)

    plt.xticks([])
    plt.yticks([])
    print("Number of samples is ", numSamples)
    for i in range(numSamples):
        # modColor = color*(1 - 0.3*np.random.rand(3))
        plt.plot(
            xProjCoords[i, :],
            yProjCoords[i, :],
            color=color,
            linewidth=1000.0 / numSamples,
            alpha=0.05,
        )

    # add a black surrounding line around all possible trajectories
    plt.plot(
        np.ones(numDims + 1) * xProjFactor,
        np.ones(numDims + 1) * yProjFactor,
        color=[0.0, 0.0, 0.0],
        alpha=0.5,
        linewidth=0.35,
    )
    return fig


def plotTest(modelName):
    """Visualize the results of EPI applied to a model with 2 parameters and
        2 output dimensions as surface plots.

    Inputs: modelName: str: model ID

    Outputs: <none>
    """

    # ---------------------------------------------------------------------------
    # First, we load and visualize the underlying data
    (
        paramDim,
        dataDim,
        numDataPoints,
        centralParam,
        data,
        dataStdevs,
    ) = dataLoader(modelName)

    # define the number of grid points per data dimension
    dataPlotResolution = 25

    # create the grid over which the data KDE will be evaluated
    dataxGrid = np.linspace(
        -0.2 * np.exp(2.0), 1.2 * np.exp(2.0), dataPlotResolution
    )
    datayGrid = np.linspace(
        -0.2 * np.exp(4.0), 1.2 * np.exp(4.0), dataPlotResolution
    )
    dataxMesh, datayMesh = np.meshgrid(dataxGrid, datayGrid)

    # allocate storage for the data density evaluation
    dataEvals = np.zeros((dataPlotResolution, dataPlotResolution))

    # double loop over all grid points of the 2D data grid
    for i in range(dataPlotResolution):
        for j in range(dataPlotResolution):
            # define the evaluation grid point
            dataEvalPoint = np.array([dataxMesh[i, j], datayMesh[i, j]])
            # evaluate the data density at the defined grid point
            dataEvals[i, j] = evalKDEGauss(data, dataEvalPoint, dataStdevs)

    # plot the data KDE using a surface plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("Data KDE")
    surf = ax.plot_surface(
        dataxMesh,
        datayMesh,
        dataEvals,
        alpha=0.95,
        cmap=cm.RdBu,
        linewidth=0,
        antialiased=False,
    )
    plt.show()

    # ---------------------------------------------------------------------------
    # Second, we load the emcee parameter sampling results and als visualize them
    paramChain = np.loadtxt(
        "Applications/" + modelName + "/OverallParams.csv", delimiter=","
    )

    # calculate reasonable standard deviations for the KDE
    paramChainStdevs = calcStdevs(paramChain)

    # define the number of grid points per parameter dimension
    paramPlotResolution = 25

    # define the grid over which the inferred parameter density will be evaluated
    paramxGrid = np.linspace(0.8, 2.2, paramPlotResolution)
    paramyGrid = np.linspace(0.8, 2.2, paramPlotResolution)

    paramxMesh, paramyMesh = np.meshgrid(paramxGrid, paramyGrid)

    # allocate storage for the parameter density evaluation
    paramEvals = np.zeros((paramPlotResolution, paramPlotResolution))

    # double loop over all grid points of the 2D parameter grid
    for i in range(paramPlotResolution):
        for j in range(paramPlotResolution):
            # define the evaluation parameter point
            paramEvalPoint = np.array([paramxMesh[i, j], paramyMesh[i, j]])
            # evaluate the parameter KDE at the defined grid point
            paramEvals[i, j] = evalKDEGauss(
                paramChain, paramEvalPoint, paramChainStdevs
            )

    # plot the inferred parameter distribution
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("Paramter Sample KDE")
    surf = ax.plot_surface(
        paramxMesh,
        paramyMesh,
        paramEvals,
        alpha=0.95,
        cmap=cm.PiYG,
        linewidth=0,
        antialiased=False,
    )
    plt.show()

    # The parameter distribution does not match the bivariate uniform distribution used to generate the data in the first place.
    # We will now show that the solution obtained by us still is one of the correct parameter distributions and therefore plot the KDE for the simulation results associated with the MCMC parameter samples
    # Unfortunately, the presented parameter inference problem is uniquely solvable.

    # ---------------------------------------------------------------------------
    # We use the same grid and colours for data and simulation results

    # Load the MCMC simulation results
    simResultsChain = np.loadtxt(
        "Applications/" + modelName + "/OverallSimResults.csv", delimiter=","
    )

    # calculate reasonable standard deviations for the KDE
    simResultsChainStdevs = calcStdevs(simResultsChain)

    # allocate storage for the data density evaluation
    simResultsEvals = np.zeros((dataPlotResolution, dataPlotResolution))

    # double loop over all grid points of the 2D data grid
    for i in range(dataPlotResolution):
        for j in range(dataPlotResolution):
            # define the evaluation grid point
            simResultsEvalPoint = np.array([dataxMesh[i, j], datayMesh[i, j]])
            # evaluate the data density at the defined grid point
            simResultsEvals[i, j] = evalKDEGauss(
                simResultsChain, simResultsEvalPoint, simResultsChainStdevs
            )

    # plot the data KDE using a surface plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("Simulation Results Sample KDE")
    surf = ax.plot_surface(
        dataxMesh,
        datayMesh,
        simResultsEvals,
        alpha=0.95,
        cmap=cm.RdBu,
        linewidth=0,
        antialiased=False,
    )
    plt.show()
