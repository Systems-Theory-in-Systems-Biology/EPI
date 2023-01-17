""" A module to quickly visualize your results from the EPI algorithm
using the matplotlib plotting library.

The flow of data is the following:

"""


from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from epic.core.kernel_density_estimation import evalKDEGauss
from epic.core.model import Model

# Colors
colorQ = np.array([255.0, 147.0, 79.0]) / 255.0
colorQApprox = np.array([204.0, 45.0, 53.0]) / 255.0
colorY = np.array([5.0, 142.0, 217.0]) / 255.0
colorYApprox = np.array([132.0, 143.0, 162.0]) / 255.0

colorExtra1 = np.array([45.0, 49.0, 66.0]) / 255.0
colorExtra2 = np.array([255.0, 218.0, 174.0]) / 255.0


def plotKDEoverGrid(
    data: np.ndarray, stdevs: np.ndarray, resolution: int = 101
) -> None:
    """Plot the 1D kernel density estimation of different data sets over a grid

    :param data: array of array of samples
    :type data: np.ndarray
    :param stdevs: one kernel standard deviation for each dimension
    :type stdevs: np.ndarray
    :param resolution: _description_, defaults to 101
    :type resolution: int, optional
    """

    for dim in range(data.shape[1]):
        minData = np.amin(data[:, dim])
        maxData = np.amax(data[:, dim])
        diffData = maxData - minData

        evalPoints = np.linspace(
            minData - 0.25 * diffData, maxData + 0.25 * diffData, resolution
        )
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


class DataParamEnum(Enum):
    Data = (0,)
    Params = 1


# Plotting:
def plotDataSamples(model: Model):
    """Scatter plot of data samples?

    :param model: _description_
    :type model: Model
    """

    artificialModel = model.isArtificial()
    name = "Sim" if artificialModel else "Meas"
    sim_measure_label = (
        "Simulation data" if artificialModel else "Measured data"
    )
    # plot data samples
    data = model.dataLoader()[4]
    for dim in data.shape[1]:
        plt.figure(figsize=(6, 1))
        plt.xlabel(r"data_dim_i")
        plt.scatter(
            data[:, dim],
            np.zeros(data.shape[0]),
            marker=r"d",
            color=colorY,
            alpha=0.1,
            label=sim_measure_label + " (Sample)",
        )
        plt.legend()
        # if not artificial: plot inferred data samples from result param samples
        # if not artificialModel:
        #     # Second, we load the emcee parameter sampling results and als visualize them
        #     simResults = model.loadSimResults(0,1)[1] # Picking simResults
        #     plt.scatter(
        #         simResults[:,dim],
        #         np.zeros(simResults.shape[0]),
        #         color=colorYApprox,
        #         alpha=0.1,
        #         label="Inferred Data samples"
        #     )
    plt.show()


def plotDataKDE(model: Model):
    """Continuos plot of data kde?

    :param model: _description_
    :type model: Model
    """
    # plot data kde
    # if not artificial: plot inferred data kde from inferred data samples
    pass


def plotParamSamples(model: Model):
    """Scatter plot of param samples?

    :param model: _description_
    :type model: Model
    """
    # if artificial: plot param samples from file
    # else: sample from inferred param distr to get samples
    pass


def plotParamKDE(model: Model):
    """Continuos plot of param kde?

    :param model: _description_
    :type model: Model
    """
    # if artificial: plot param kde from param samples file
    # plot inferred param kde from paramChain sim results
    pass


def sampleFromResults(model: Model, sampleSize: int = 1000):
    """Samples from the calculated param distribution
    and calculates the data points for these parameters

    :param model: The model from which the results shall be loaded
    :type model: Model
    :param sampleSize: number of drawn samples, defaults to 1000
    :type sampleSize: int, optional
    """
    pass
