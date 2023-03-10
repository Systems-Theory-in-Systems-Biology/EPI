from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit

from epi import logger
from epi.core.kde import eval_kde_gauss
from epi.core.model import Model


def evaluate_density(
    param: np.ndarray,
    model: Model,
    data: np.ndarray,
    dataStdevs: np.ndarray,
    slice: np.ndarray,
) -> Tuple[np.double, np.ndarray]:
    """Given a simulation model, its derivative and corresponding data, evaluate the parameter density that is the backtransformed data distribution.

    Args:
        param (np.ndarray): parameter for which the transformed density shall be evaluated
        model (Model): model to be evaluated
        data (np.ndarray): data for the model. 2D array with shape (#num_data_points, #data_dim)
        dataStdevs (np.ndarray): array of suitable kernel width for each data dimension
        slice (np.ndarray): slice of the parameter vector that is to be evaluated

    Returns:
        Tuple[np.double, np.ndarray]:
            : parameter density at the point param
            : vector containing the parameter, the simulation result and the density
    """

    limits = model.param_limits

    # Build the full parameter vector for evaluation based on the passed param slice and the constant central points
    fullParam = model.central_param
    fullParam[slice] = param

    # Check if the tried parameter is within the just-defined bounds and return the lowest possible density if not.
    if np.any((param < limits[slice, 0]) | (param > limits[slice, 1])):
        logger.info(
            "Parameters outside of predefined range"
        )  # Slows down the sampling to much? -> Change logger level to warning or even error
        return 0, np.zeros(slice.shape[0] + model.data_dim + 1)

    # If the parameter is within the valid ranges...
    else:
        # Evaluating the model and the jacobian for the specified parameter simultaneously provide a little speedup over calculating it separately in some cases.
        sim_res, jac = model.valjac(fullParam)

        # Evaluate the data density in the simulation result.
        densityEvaluation = eval_kde_gauss(data, sim_res, dataStdevs)

        # Calculate the simulation model's pseudo-determinant in the parameter point (also called the correction factor).
        correction = calc_gram_determinant(jac)

        # Multiply data density and correction factor.
        trafo_density_evaluation = densityEvaluation * correction

        # Store the current parameter, its simulation result as well as its density in a large vector that is stored separately by emcee.
        evaluation_results = np.concatenate(
            (param, sim_res, np.array([trafo_density_evaluation]))
        )

        return trafo_density_evaluation, evaluation_results


def eval_log_transformed_density(
    param: np.ndarray,
    model: Model,
    data: np.ndarray,
    dataStdevs: np.ndarray,
    slice: np.ndarray,
) -> Tuple[np.double, np.ndarray]:
    """Given a simulation model, its derivative and corresponding data, evaluate the natural log of the parameter density that is the backtransformed data distribution.
        This function is intended to be used with the emcee sampler and can be implemented more efficiently at some points.

    Args:
        param (np.ndarray): parameter for which the transformed density shall be evaluated
        model (Model): model to be evaluated
        data (np.ndarray): data for the model. 2D array with shape (#num_data_points, #data_dim)
        dataStdevs (np.ndarray): array of suitable kernel width for each data dimension
        slice (np.ndarray): slice of the parameter vector that is to be evaluated

    Returns:
        Tuple[np.double, np.ndarray]:
            : natural log of the parameter density at the point param
            : sampler_results (array concatenation of parameters, simulation results and evaluated density, stored as "blob" by the emcee sampler)

    """
    trafo_density_evaluation, evaluation_results = evaluate_density(
        param, model, data, dataStdevs, slice
    )
    if trafo_density_evaluation == 0:
        return -np.inf, evaluation_results
    return np.log(trafo_density_evaluation), evaluation_results


def calc_gram_determinant(jac: jnp.ndarray) -> jnp.double:
    """Evaluate the pseudo-determinant of the jacobian (that serves as a correction term) in one specific parameter point.
    Returns 0 if the correction factor is not finite.

    Args:
      jac(jnp.ndarray): The jacobian for which the pseudo determinant shall be calculated

    Returns:
        jnp.double: The pseudo-determinant of the jacobian

    """
    correction = _calc_gram_determinant(jac)
    # If the correction factor is not finite, return 0 instead to not affect the sampling.
    if not jnp.isfinite(correction):
        correction = 0.0
        logger.warning("Invalid value encountered for correction factor")
    return correction


@jit
def _calc_gram_determinant(jac: jnp.ndarray) -> jnp.double:
    """Jitted calculation of the pseudo-determinant of the jacobian. This function is called by calc_gram_determinant() and should not be called directly.
    It does not check if the correction factor is finite.

    Not much faster than a similar numpy version. However it can run on gpu and is maybe a bit faster because we can jit compile the sequence of operations.

    Args:
        jac (jnp.ndarray): The jacobian for which the pseudo determinant shall be calculated

    Returns:
        jnp.double: The pseudo-determinant of the jacobian

    """

    jac = jnp.atleast_2d(jac)

    if jac.shape[0] == jac.shape[1]:
        return jnp.abs(jnp.linalg.det(jac))
    else:
        jacT = jnp.transpose(jac)
        # The pseudo-determinant is calculated as the square root of the determinant of the matrix-product of the Jacobian and its transpose.
        # For numerical reasons, one can regularize the matrix product by adding a diagonal matrix of ones before calculating the determinant.
        # correction = np.sqrt(np.linalg.det(np.matmul(jacT,jac) + np.eye(param.shape[0])))
        correction = jnp.sqrt(jnp.linalg.det(jnp.matmul(jacT, jac)))
        return correction
