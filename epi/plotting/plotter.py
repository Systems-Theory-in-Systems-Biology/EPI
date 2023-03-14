""" A module to quickly visualize your results from the EPI algorithm
using the matplotlib plotting library.
"""


from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from epi.core.kde import eval_kde_gauss
from epi.core.model import Model

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

    Args:
      data(np.ndarray): array of array of samples
      stdevs(np.ndarray): one kernel standard deviation for each dimension
      resolution(int, optional): _description_, defaults to 101
      data: np.ndarray:
      stdevs: np.ndarray:
      resolution: int:  (Default value = 101)

    Returns:

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
            evaluations[i] = eval_kde_gauss(
                np.transpose(np.array([data[:, dim]])),
                np.array([evalPoints[i]]),
                np.array([stdevs[dim]]),
            )

        plt.figure()
        plt.plot(evalPoints, evaluations)
        plt.hist(data[:, dim], bins=evalPoints, density=True)
        plt.show()


class DataParamEnum(Enum):
    """ """

    Data = (0,)
    Params = 1


# Plotting:
def plotDataSamples(model: Model):
    """Scatter plot of data samples?

    Args:
      model(Model): _description_
      model: Model:

    Returns:

    """

    artificialModel = model.is_artificial()
    name = "Sim" if artificialModel else "Meas"
    sim_measure_label = (
        "Simulation data" if artificialModel else "Measured data"
    )
    # plot data samples
    data = model.dataLoader()[3]
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
        #     sim_results = model.load_sim_results(0,1)[1] # Picking sim_results
        #     plt.scatter(
        #         sim_results[:,dim],
        #         np.zeros(sim_results.shape[0]),
        #         color=colorYApprox,
        #         alpha=0.1,
        #         label="Inferred Data samples"
        #     )
    plt.show()


def plotDataKDE(model: Model):
    """Continuos plot of data kde?

    Args:
      model(Model): _description_
      model: Model:

    Returns:

    """
    # plot data kde
    # if not artificial: plot inferred data kde from inferred data samples
    pass


def plotParamSamples(model: Model):
    """Scatter plot of param samples?

    Args:
      model(Model): _description_
      model: Model:

    Returns:

    """
    # if artificial: plot param samples from file
    # else: sample from inferred param distr to get samples
    pass


def plotParamKDE(model: Model):
    """Continuos plot of param kde?

    Args:
      model(Model): _description_
      model: Model:

    Returns:

    """
    # if artificial: plot param kde from param samples file
    # plot inferred param kde from param_chain sim results
    pass


def sampleFromResults(model: Model, sampleSize: int = 1000):
    """Samples from the calculated param distribution
    and calculates the data points for these parameters

    Args:
      model(Model): The model from which the results shall be loaded
      sampleSize(int, optional): number of drawn samples, defaults to 1000
      model: Model:
      sampleSize: int:  (Default value = 1000)

    Returns:

    """
    pass
