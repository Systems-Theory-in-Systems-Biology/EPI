import numpy as np
from scipy.stats import cauchy, norm


def evalKDECauchy_fast(data, simRes, scales):
    return (
        np.sum(np.prod(cauchy.pdf(data, simRes, scales), axis=1))
        / data.shape[0]
    )


def evalKDEGauss_fast(data, simRes, scales):
    return (
        np.sum(np.prod(norm.pdf(data, simRes, scales), axis=1)) / data.shape[0]
    )


def evalKDECauchy(data, simRes, scales):
    r"""Evaluates a Cauchy Kernel Density estimator in one simulation result.
        Assumes that each data point is a potentially high-dimensional sample
        from a joint data distribution.
        This is for example given for time-series data, where each evaluation
        time is one dimension of the data point.

        .. math::
            density = \\frac{1}{samples} \sum_{s=1}^{samples} \prod_{d=1}^{dims} \\frac{1}{(\\frac{x^{eval}_d - x_{s,d}}{scales_d})^2 \; \pi \; scales_d}

    Input: data (data for the model: 2D array with shape (#Samples, #MeasurementDimensions))
           simRes (evaluation coordinates array with one entry for each data dimension)
           scales (one scale for each dimension)

    :return: densityEvaluation (estimated kernel density evaluated at the simulation result)
    """
    # This quantity will store the probability density.
    evaluation = 0

    # Loop over each measurement sample.
    for s in range(data.shape[0]):
        # Construct a Cauchy-ditribution centered around the data point and evaluate it in the simulation result.
        evaluation += np.prod(
            1
            / (
                (np.power((simRes - data[s, :]) / scales, 2) + 1)
                * scales
                * np.pi
            )
        )

    # Return the average of all Cauchy distribution evaluations to eventually obtain a probability density again.
    return evaluation / data.shape[0]


def evalKDEGauss(data, simRes, stdevs):
    """Evaluates a Gaussian Kernel Density estimator in one simulation result.
        Assumes that each data point is a potentially high-dimensional sample from a joint data distribution.
        This is for example given for time-series data, where each evaluation time is one dimension of the data point.
        While it is possible to define different standard deviations for different measurement dimensions, it is so far not possible to define covariances.

    Input: data (data for the model: 2D array with shape (#Samples, #MeasurementDimensions))
           simRes (evaluation coordinates array with one entry for each data dimension)
           stdevs (one standard deviation for each dimension)

    Output: densityEvaluation (estimated kernel density evaluated at the simulation result)
    """
    # This quantity will store the probability density
    evaluation = 0

    # Loop over each measurement sample
    for s in range(data.shape[0]):
        # Construct a Cauchy-ditribution centered around the data point and evaluate it in the simulation result.
        diff = simRes - data[s, :]
        mult = -np.sum(diff * diff / stdevs / stdevs) / 2.0

        evaluation += np.exp(mult) / np.sqrt(
            np.power(2 * np.pi, simRes.shape[0]) * np.power(np.prod(stdevs), 2)
        )

    # Return the average of all Gauss distribution evaluations to eventually obtain a probability density again.
    return evaluation / data.shape[0]


def calcKernelWidth(data):
    """Sets the width of the kernels used for density estimation of the data according to the Silverman rule

    Input: data: 2d array with shape (#Samples, #MeasurementDimensions): data for the model

    Output: stdevs: array with shape (#MeasurementDimensions): suitable kernel standard deviations for each measurement dimension
    """

    numDataPoints, dataDim = data.shape

    means = np.mean(data, axis=0)
    stdevs = np.std(data, axis=0, ddof=1)
    maxStderiv = np.amax(stdevs)

    # Silvermans rule
    return stdevs * (numDataPoints * (dataDim + 2) / 4.0) ** (
        -1.0 / (dataDim + 4)
    )
