# Imports

import diffrax as dx
import jax.numpy as jnp
import numpy as np
import yfinance as yf
from jax import jacrev  # , grad, jacfwd
from jax.config import config

config.update("jax_enable_x64", True)
# import datetime
# import pickle
# import pprint
# from functools import partial
# import arviz
# import corner
# import pandas as pd
# import scipy as sp
# from matplotlib import cm
# from scipy import stats
# import matplotlib.pyplot as plt
# from jax.experimental.ode import odeint


def returnVisualizationGrid(modelName, resolution):
    """
        Specify the plot grids for each model's parameters and data.
        The grids are stored as files.

    Inputs: modelName: str: model ID
            resolution: int: number of grid points for every dimension

    Outputs: <none>
    """

    # load the model data and characteristics
    (
        paramDim,
        dataDim,
        numDataPoints,
        centralParam,
        data,
        dataStdevs,
    ) = dataLoader(modelName)

    # allocate storage for the parameter and data plotting grid
    paramGrid = np.zeros((resolution, paramDim))
    dataGrid = np.zeros((resolution, dataDim))

    # define specific boundaries for each model
    if (modelName == "Corona") or (modelName == "CoronaArtificial"):
        dataGrid[:, 0] = np.linspace(0.0, 4.0, resolution)
        dataGrid[:, 1] = np.linspace(0.0, 40.0, resolution)
        dataGrid[:, 2] = np.linspace(0.0, 80.0, resolution)
        dataGrid[:, 3] = np.linspace(0.0, 3.5, resolution)

        paramGrid[:, 0] = np.linspace(-4.0, 0.0, resolution)
        paramGrid[:, 1] = np.linspace(-2.0, 2.0, resolution)
        paramGrid[:, 2] = np.linspace(-1.0, 3.0, resolution)

    elif (modelName == "Stock") or (modelName == "StockArtificial"):
        for i in range(dataDim):
            dataGrid[:, i] = np.linspace(-7.5, 7.5, resolution)

        for i in range(paramDim):
            paramGrid[:, i] = np.linspace(-2.0, 2.0, resolution)

    # raise a warning if the model is misspecified
    else:
        print("No grid specified for this model!")

    # store both grids as csv files into the model-specific plot directory
    np.savetxt(
        "Applications/" + modelName + "/Plots/dataGrid.csv",
        dataGrid,
        delimiter=",",
    )
    np.savetxt(
        "Applications/" + modelName + "/Plots/paramGrid.csv",
        paramGrid,
        delimiter=",",
    )

    return 0


def calcStdevs(data):
    """Sets the width of the kernels used for density estimation of the data according to the Silverman rule

    Input: data: 2d array with shape (#Samples, #MeasurementDimensions): data for the model

    Output: stdevs: array with shape (#MeasurementDimensions): suitable kernel standard deviations for each measurement dimension
    """

    numDataPoints, dataDim = data.shape

    means = np.sum(data, 0) / numDataPoints
    stdevs = np.zeros(dataDim)

    for i in range(dataDim):
        stdevs[i] = np.sqrt(
            np.sum(np.power(data[:, i] - means[i], 2)) / (numDataPoints - 1)
        )

    maxStderiv = np.amax(stdevs)

    # Silvermans rule
    return stdevs * (numDataPoints * (dataDim + 2) / 4.0) ** (
        -1.0 / (dataDim + 4)
    )


def dataLoader(modelName):
    """loads and returns all data for a chosen model

    Args:
        modelName (_type_): The model id

    Returns:
        _type_: paramDim (number of model parameters)
                dataDim (number of model output dimensions)
                numDataPoints (number of data points)
                centralParam (meaningful initial parameter set for the defined model)
                data (data for the model: 2D array with shape (#numDataPoints, #dataDim))
                dataStdevs (array of suitable kernel standard deviations for each data dimension)
    """
    centralParamsDict = {
        "Linear": np.array([0.5, 0.5]),
        "LinearODE": np.array([1.5, 1.5]),
        "Temperature": np.array([np.pi / 4.0]),
        "TemperatureArtificial": np.array([np.pi / 4.0]),
        "Corona": np.array([-1.8, 0.0, 0.7]),
        "CoronaArtificial": np.array([-1.8, 0.0, 0.7]),
        "Stock": np.array(
            [
                0.41406223,
                1.04680993,
                1.21173553,
                0.8078955,
                1.07772437,
                0.64869251,
            ]
        ),
        "StockArtificial": np.array(
            [
                0.41406223,
                1.04680993,
                1.21173553,
                0.8078955,
                1.07772437,
                0.64869251,
            ]
        ),
    }

    centralParam = centralParamsDict[modelName]

    paramDim = centralParam.shape[0]

    data = np.loadtxt("Data/" + modelName + "Data.csv", delimiter=",")

    if len(data.shape) == 1:
        data = data.reshape((data.shape[0], 1))

    dataStdevs = calcStdevs(data)

    numDataPoints, dataDim = data.shape

    return paramDim, dataDim, numDataPoints, centralParam, data, dataStdevs


def paramLoader(modelName):
    """loads and returns all parameters for artificial set ups

    Args:
        modelName (_type_): Model ID

    Returns:
        _type_: params (true parameters used to generate artificial data)
                paramStdevs (array of suitable kernel standard deviations for each parameter dimension)
    """
    trueParams = np.loadtxt("Data/" + modelName + "Params.csv", delimiter=",")

    if len(trueParams.shape) == 1:
        trueParams = trueParams.reshape((trueParams.shape[0], 1))

    paramStdevs = calcStdevs(trueParams)

    return trueParams, paramStdevs


def modelLoader(modelName):
    """loads and returns the model and its jacobian

    Args:
        modelName (_type_): Model ID

    Returns:
        _type_: model (simulation model)
                modelJac (algorithmically differentiated simulation model)
    """
    if (modelName == "Temperature") or (modelName == "TemperatureArtificial"):
        model = temperatureModel
        modelJac = jacrev(model)

    elif (modelName == "Corona") or (modelName == "CoronaArtificial"):
        model = coronaModel
        modelJac = jacrev(model)

    elif (modelName == "Stock") or (modelName == "StockArtificial"):
        model = stockModel
        modelJac = jacrev(model)

    elif modelName == "Linear":
        model = linModel
        modelJac = jacrev(model)

    elif modelName == "Exponential":
        model = expModel
        modelJac = jacrev(model)

    elif modelName == "LinearODE":
        model = linODEModel
        modelJac = jacrev(model)
    else:
        print("Invalid Model choice!")

    return model, modelJac


# Function that generate artificial data sets using the models defined below


def generateLinearData():
    numSamples = 1000

    # randomly create true parameters in [0,1]^2
    trueParamSample = np.random.rand(numSamples, 2)

    artificialData = np.zeros((trueParamSample.shape[0], 2))

    for i in range(trueParamSample.shape[0]):
        artificialData[i, :] = linModel(trueParamSample[i, :])

    np.savetxt("Data/LinearData.csv", artificialData, delimiter=",")
    np.savetxt("Data/LinearParams.csv", trueParamSample, delimiter=",")

    return 0


def generateLinearODEData():
    numSamples = 1000

    # randomly create true parameters in [1,2]^2
    trueParamSample = np.random.rand(numSamples, 2) + 1

    artificialData = np.zeros((trueParamSample.shape[0], 2))

    for i in range(trueParamSample.shape[0]):
        artificialData[i, :] = linODEModel(trueParamSample[i, :])

    np.savetxt("Data/LinearODEData.csv", artificialData, delimiter=",")
    np.savetxt("Data/LinearODEParams.csv", trueParamSample, delimiter=",")

    return 0


def generateArtificialTemperatureData():
    rawTrueParamSample = np.loadtxt(
        "Data/TemperatureArtificialParams.csv", delimiter=","
    )
    trueParamSample = np.zeros((rawTrueParamSample.shape[0], 1))
    trueParamSample[:, 0] = rawTrueParamSample

    artificialData = np.zeros((trueParamSample.shape[0], 1))

    for i in range(trueParamSample.shape[0]):
        artificialData[i, 0] = temperatureModel(trueParamSample[i, :])

    np.savetxt(
        "Data/TemperatureArtificialData.csv", artificialData, delimiter=","
    )

    return 0


def generateArtificialCoronaData():
    numSamples = 10000

    lowerBound = np.array([-1.9, -0.1, 0.6])
    upperBound = np.array([-1.7, 0.1, 0.8])

    trueParamSample = np.random.rand(numSamples, 3)

    artificialData = np.zeros((numSamples, 4))

    for j in range(numSamples):
        trueParamSample[j, :] = (
            lowerBound + (upperBound - lowerBound) * trueParamSample[j, :]
        )
        artificialData[j, :] = coronaModel(trueParamSample[j, :])

    np.savetxt("Data/CoronaArtificialData.csv", artificialData, delimiter=",")
    np.savetxt(
        "Data/CoronaArtificialParams.csv", trueParamSample, delimiter=","
    )

    return 0


def generateArtificialStockData():
    numSamples = 100000

    mean = np.array(
        [0.41406223, 1.04680993, 1.21173553, 0.8078955, 1.07772437, 0.64869251]
    )
    stdevs = np.array([0.005, 0.01, 0.05, 0.005, 0.01, 0.05])

    trueParamSample = np.random.randn(numSamples, 6)

    for i in range(6):
        trueParamSample[:, i] *= stdevs[i]

    artificialData = np.zeros((numSamples, 19))

    for j in range(numSamples):
        trueParamSample[j, :] += mean
        artificialData[j, :] = stockModel(trueParamSample[j, :])

    np.savetxt("Data/StockArtificialData.csv", artificialData, delimiter=",")
    np.savetxt(
        "Data/StockArtificialParams.csv", trueParamSample, delimiter=","
    )

    return 0


# Downloads data for over 6000 Stocks and extracts their value according to indicated time points
def generateStockData(tickerList, tickerListName):
    # define

    start = "2022-01-31"
    end = "2022-03-01"
    dates = np.array(
        [
            "2022-01-31",
            "2022-02-01",
            "2022-02-02",
            "2022-02-03",
            "2022-02-04",
            "2022-02-07",
            "2022-02-08",
            "2022-02-09",
            "2022-02-10",
            "2022-02-11",
            "2022-02-14",
            "2022-02-15",
            "2022-02-16",
            "2022-02-17",
            "2022-02-18",
            "2022-02-22",
            "2022-02-23",
            "2022-02-24",
            "2022-02-25",
            "2022-02-28",
        ]
    )

    stocks = np.loadtxt(tickerList, dtype="str")

    stockData = np.zeros((stocks.shape[0], dates.shape[0]))
    stockIDs = []

    successCounter = 0

    for i in range(stocks.shape[0]):
        try:
            df = yf.download(stocks[i], start, end, interval="1d")

            try:
                for j in range(dates.shape[0]):
                    # extract the opening value (indicated by "[0]") of stock i at day j
                    stockData[successCounter, j] = df.loc[str(dates[j])][0]

                # subtract initial value of complete timeline
                stockData[successCounter, :] = (
                    stockData[successCounter, :] - stockData[successCounter, 0]
                )

                if np.all(np.abs(stockData[successCounter, :]) < 100.0):
                    print("Success for ", stocks[i])
                    stockIDs.append(stocks[i])
                    successCounter += 1
                else:
                    print("Values too large for ", stocks[i])

            except Exception as e:
                print("Fail for ", stocks[i])
                print(repr(e))
                pass

        except Exception as e:
            print("Download Failed!")
            print(repr(e))

    # save all time points except for the first
    np.savetxt(
        "Data/DataOrigins/Stock/" + tickerListName + "Data.csv",
        stockData[0:successCounter, 1:],
        delimiter=",",
    )
    np.savetxt(
        "Data/DataOrigins/Stock/" + tickerListName + "IDs.csv",
        stockIDs,
        delimiter=",",
        fmt="% s",
    )

    return 0


# Models corresponding to previously defined data sets
# Input: array of parameters
# Output: array of model results with one entry per corresponding measurement dimension

# Model for Latitude-dependent annual average temperature
def temperatureModel(param):
    lowT = -30.0
    highT = 30.0

    res = jnp.array([lowT + (highT - lowT) * jnp.cos(jnp.abs(param[0]))])

    return res


# SEIR Model for regional, weekly Corona incidence development with fewer time points after 1,2,5, and 15 weeks, respectively


def coronaModel(logParam):
    param = jnp.power(10, logParam)
    xInit = jnp.array([999.0, 0.0, 1.0, 0.0])

    def rhs(t, x, param):
        return jnp.array(
            [
                -param[0] * x[0] * x[2],
                param[0] * x[0] * x[2] - param[1] * x[1],
                param[1] * x[1] - param[2] * x[2],
                param[2] * x[2],
            ]
        )

    term = dx.ODETerm(rhs)
    solver = dx.Kvaerno5()
    saveat = dx.SaveAt(ts=[0.0, 1.0, 2.0, 5.0, 15.0])
    stepsize_controller = dx.PIDController(rtol=1e-5, atol=1e-5)

    try:
        odeSol = dx.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=15.0,
            dt0=0.1,
            y0=xInit,
            args=param,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
        )
        return odeSol.ys[1:5, 2]

    except Exception as e:
        print("ODE solution not possible!")
        print(repr(e))
        return np.array([-np.inf, -np.inf, -np.inf, -np.inf])


# Model for stock price development


def stockModel(param):
    def iteration(x, param):
        return jnp.array(
            [
                param[0] / (1 + jnp.power(x[0], 2))
                + param[1] * x[1]
                + param[2],
                param[3] * x[1] - param[4] * x[0] + param[5],
            ]
        )

    def repetition(x, param, numRepetitions):
        for i in range(numRepetitions):
            x = iteration(x, param)

        return x

    x0 = jnp.zeros(2)
    x1 = repetition(x0, param, 1)
    x2 = repetition(x1, param, 1)
    x3 = repetition(x2, param, 1)
    x4 = repetition(x3, param, 1)
    x5 = repetition(x4, param, 3)
    x6 = repetition(x5, param, 1)
    x7 = repetition(x6, param, 1)
    x8 = repetition(x7, param, 1)
    x9 = repetition(x8, param, 1)
    x10 = repetition(x9, param, 3)
    x11 = repetition(x10, param, 1)
    x12 = repetition(x11, param, 1)
    x13 = repetition(x12, param, 1)
    x14 = repetition(x13, param, 1)
    x15 = repetition(x14, param, 4)
    x16 = repetition(x15, param, 1)
    x17 = repetition(x16, param, 1)
    x18 = repetition(x17, param, 1)
    x19 = repetition(x18, param, 3)

    timeCourse = jnp.array(
        [
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            x8,
            x9,
            x10,
            x11,
            x12,
            x13,
            x14,
            x15,
            x16,
            x17,
            x18,
            x19,
        ]
    )

    return timeCourse[:, 0]


# very simple linear test model: [0,1]^2 -> [0,10]x[-2,-4]
def linModel(param):
    return jnp.array([param[0] * 10, (-2.0) * param[1] - 2.0])


# very simple exponential test model
def expModel(param):
    return jnp.array([param[0] * jnp.exp(1), jnp.exp(param[1])])


# very simple linear ODE test model a*e^(bt)
def linODEModel(param):
    return jnp.array(
        [
            param[0] * jnp.exp(param[1] * 1.0),
            param[0] * jnp.exp(param[1] * 2.0),
        ]
    )
