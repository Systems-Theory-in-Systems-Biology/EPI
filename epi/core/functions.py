import numpy as np

from epi import logger
from epi.core.kernel_density_estimation import evalKDEGauss
from epi.core.model import Model


def evalLogTransformedDensity(
    param: np.ndarray, model: Model, data: np.ndarray, dataStdevs: np.ndarray
) -> tuple[np.double, np.ndarray]:
    """Given a simulation model, its derivative and corresponding data, evaluate the natural log of the parameter density that is the backtransformed data distribution.
        This function is intended to be used with the emcee sampler and can be implemented more efficiently at some points.

    Input: param (parameter for which the transformed density shall be evaluated)
           model
           data (data for the model: 2D array with shape (#numDataPoints, #dataDim))
           dataStdevs (array of suitable kernel standard deviations for each data dimension)
    Output: logTransformedDensity (natural log of parameter density at the point param)
          : allRes (array concatenation of parameters, simulation results and evaluated density, stored as "blob" by the emcee sampler)
    """
    limits = model.getParamSamplingLimits()

    # Check if the tried parameter is within the just-defined bounds and return the lowest possible log density if not.
    if np.any((param < limits[:, 0]) | (param > limits[:, 1])):
        logger.info(
            "Parameters outside of predefined range"
        )  # Slows down the sampling to much? -> Change logger level to warning or even error
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
