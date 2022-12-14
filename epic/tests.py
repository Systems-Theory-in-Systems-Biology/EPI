import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from epic.functions import evalKDECauchy, evalKDEGauss, evalLogTransformedDensity
from epic.models import (
    calcKernelWidth,
    generateArtificialCoronaData,
    generateArtificialStockData,
    generateArtificialTemperatureData,
    generateLinearData,
    generateLinearODEData,
    generateStockData,
    modelLoader,
)
from epic.plots import plotTest
from epic.sampling import concatenateEmceeSamplingResults, runEmceeSampling

# test both kernel density estimator by using one data point and evaluating the Kernel Density Estimation over a grid


def KDETest1DataPoint():
    # define the one data point
    data = np.array([[0.5, 2.0]])

    # define kernel standard deviations
    stdevs = np.array([2.0, 1.0])

    # number of test grid points
    resolution = 33

    xGrid = np.linspace(-2.5, 3.5, resolution)
    yGrid = np.linspace(0.5, 3.5, resolution)

    xMesh, yMesh = np.meshgrid(xGrid, yGrid)

    gaussEvals = np.zeros((resolution, resolution))
    cauchyEvals = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            evalPoint = np.array([xMesh[i, j], yMesh[i, j]])

            gaussEvals[i, j] = evalKDEGauss(data, evalPoint, stdevs)
            cauchyEvals[i, j] = evalKDECauchy(data, evalPoint, stdevs)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xMesh,
        yMesh,
        gaussEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xMesh,
        yMesh,
        cauchyEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()


def KDETest3DataPoints():
    # define the one data point
    data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])

    # define kernel standard deviations
    stdevs = np.array([0.5, 1.0])

    # number of test grid points
    resolution = 33

    xGrid = np.linspace(-1.0, 3.0, resolution)
    yGrid = np.linspace(-1.0, 2.0, resolution)

    xMesh, yMesh = np.meshgrid(xGrid, yGrid)

    gaussEvals = np.zeros((resolution, resolution))
    cauchyEvals = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            evalPoint = np.array([xMesh[i, j], yMesh[i, j]])

            gaussEvals[i, j] = evalKDEGauss(data, evalPoint, stdevs)
            cauchyEvals[i, j] = evalKDECauchy(data, evalPoint, stdevs)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xMesh,
        yMesh,
        gaussEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xMesh,
        yMesh,
        cauchyEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()


def transformationTestLinear():
    modelName = "Linear"

    # create approx. 1000 data points that are perfectly uniformly distributed over a grid
    # the range of data points is the 2D interval [0,10]x[-2,-4]
    dataResolution = 33

    xGrid = np.linspace(0.0, 10.0, dataResolution)
    yGrid = np.linspace(-4.0, -2.0, dataResolution)
    xMesh, yMesh = np.meshgrid(xGrid, yGrid)

    data = np.array([xMesh.flatten(), yMesh.flatten()]).T

    # define standard deviations according to silverman
    dataStdevs = calcKernelWidth(data)
    print("Data standard deviations = ", dataStdevs)

    # Now plot the data Gaussian KDE
    KDEresolution = 25

    # the KDE Grid is 40% larger than the intervalof the data and has a different resolution
    KDExGrid = np.linspace(-2.0, 12.0, KDEresolution)
    KDEyGrid = np.linspace(-4.4, -1.6, KDEresolution)
    KDExMesh, KDEyMesh = np.meshgrid(KDExGrid, KDEyGrid)

    gaussEvals = np.zeros((KDEresolution, KDEresolution))

    for i in range(KDEresolution):
        for j in range(KDEresolution):
            evalPoint = np.array([KDExMesh[i, j], KDEyMesh[i, j]])
            gaussEvals[i, j] = evalKDEGauss(data, evalPoint, dataStdevs)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("Data KDE")
    surf = ax.plot_surface(
        KDExMesh,
        KDEyMesh,
        gaussEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()

    # load the linear test model and its algorithmic differentiation
    model, modelJac = modelLoader(modelName)

    paramResolution = 15
    paramxGrid = np.linspace(-0.2, 1.2, paramResolution)
    paramyGrid = np.linspace(-0.2, 1.2, paramResolution)

    paramxMesh, paramyMesh = np.meshgrid(paramxGrid, paramyGrid)

    paramEvals = np.zeros((paramResolution, paramResolution))

    for i in range(paramResolution):
        for j in range(paramResolution):
            paramPoint = np.array([paramxMesh[i, j], paramyMesh[i, j]])
            paramEvals[i, j], _ = evalLogTransformedDensity(
                paramPoint, modelName, data, dataStdevs
            )

    paramEvals = np.exp(paramEvals)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("Paramter Density Estimation")
    surf = ax.plot_surface(
        paramxMesh,
        paramyMesh,
        paramEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()


def transformationTestExponential():
    modelName = "Exponential"

    # load the exponential test model and its algorithmic differentiation
    model, modelJac = modelLoader(modelName)

    # create true parameter points that are drawn uniformly from [0,1]^2

    numDataPoints = 10000
    trueParam = np.random.rand(numDataPoints, 2) + 1

    data = np.zeros((numDataPoints, 2))

    # transform the parameter points using the model
    for i in range(numDataPoints):
        data[i, :] = model(trueParam[i, :])

    # define standard deviations according to silverman
    dataStdevs = calcKernelWidth(data)
    print("Data standard deviations = ", dataStdevs)

    # Now plot the data Gaussian KDE
    KDEresolution = 25

    KDExGrid = np.linspace(0.8 * np.exp(1), 2.2 * np.exp(1), KDEresolution)
    KDEyGrid = np.linspace(np.exp(0.8), np.exp(2.2), KDEresolution)
    KDExMesh, KDEyMesh = np.meshgrid(KDExGrid, KDEyGrid)

    gaussEvals = np.zeros((KDEresolution, KDEresolution))

    for i in range(KDEresolution):
        for j in range(KDEresolution):
            evalPoint = np.array([KDExMesh[i, j], KDEyMesh[i, j]])
            gaussEvals[i, j] = evalKDEGauss(data, evalPoint, dataStdevs)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("Data KDE")
    surf = ax.plot_surface(
        KDExMesh,
        KDEyMesh,
        gaussEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()

    paramResolution = 25
    paramxGrid = np.linspace(0.8, 2.2, paramResolution)
    paramyGrid = np.linspace(0.8, 2.2, paramResolution)

    paramxMesh, paramyMesh = np.meshgrid(paramxGrid, paramyGrid)

    paramEvals = np.zeros((paramResolution, paramResolution))

    for i in range(paramResolution):
        for j in range(paramResolution):
            paramPoint = np.array([paramxMesh[i, j], paramyMesh[i, j]])
            paramEvals[i, j], _ = evalLogTransformedDensity(
                paramPoint, modelName, data, dataStdevs
            )

    paramEvals = np.exp(paramEvals)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("Paramter Density Estimation")
    surf = ax.plot_surface(
        paramxMesh,
        paramyMesh,
        paramEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()


def transformationTestODELinear():
    """
    Exemplary application of Eulerian Parameter Inference a very simplistic ordinary differential equation model.
    We create our toy data by first defining a true parameter distribution.
    The actual data is then obtained by evaluating the model in parameters drawn from this true distribution.
    Ideally, we would be able to reconstruct the true parameter density.
    However, we will see that this parameter inference problem posesses more than one solution and is therefore not well-posed.
    """

    # define the model
    modelName = "LinearODE"

    # generate artificial data
    generateLinearODEData()

    # choose the number of subsequent runs
    # after each sub-run, chains are saved
    numRuns = 2

    # choose how many parallel processes can be used
    numProcesses = 4

    # choose how many parallel chains shall be initiated
    numWalkers = 4

    # choose how many steps each chain shall contain
    numSteps = 2500

    # run MCMC sampling for EPI
    runEmceeSampling(modelName, numRuns, numWalkers, numSteps, numProcesses)

    # combine all intermediate saves to create one large sample chain
    concatenateEmceeSamplingResults(modelName)

    # plot the results
    plotTest(modelName)
