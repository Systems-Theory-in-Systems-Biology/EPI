"""This module implements the random variable transformation of the EPI algorithm.
"""

from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit

from eulerpi import logger
from eulerpi.core.data_transformation import DataTransformation
from eulerpi.core.kde import eval_kde_gauss
from eulerpi.core.model import Model


def evaluate_density(
    param: np.ndarray,
    model: Model,
    data: np.ndarray,
    data_transformation: DataTransformation,
    data_stdevs: np.ndarray,
    slice: np.ndarray,
) -> Tuple[np.double, np.ndarray]:
    """Calculate the parameter density as backtransformed data density using the simulation model

    .. math::

        \\Phi_\\mathcal{Q}(q) = \\Phi_\\mathcal{Y}(s(q)) \\cdot \\sqrt{\\det \\left({\\frac{ds}{dq}(q)}^\\intercal {\\frac{ds}{dq}(q)}\\right)}

    Args:
        param (np.ndarray): parameter for which the transformed density shall be evaluated
        model (Model): model to be evaluated
        data (np.ndarray): data for the model. 2D array with shape (#num_data_points, #data_dim)
        data_transformation (DataTransformation): The data transformation used to normalize the data.
        data_stdevs (np.ndarray): array of suitable kernel width for each data dimension
        slice (np.ndarray): slice of the parameter vector that is to be evaluated

    Returns:
        Tuple[np.double, np.ndarray]:
            : parameter density at the point param
            : vector containing the parameter, the simulation result and the density

    Examples:

    .. code-block:: python

        import numpy as np
        from eulerpi.examples.heat import Heat
        from eulerpi.core.kde import calc_kernel_width
        from eulerpi.core.data_transformation import DataIdentity
        from eulerpi.core.transformations import evaluate_density

        # use the heat model
        model = Heat()

        # generate 1000 artificial, 5D data points for the Heat example model
        data_mean = np.array([0.5, 0.1, 0.5, 0.9, 0.5])
        data = np.random.randn(1000, 5)/25.0 + data_mean

        # evaluating the parameter probabiltiy density at the central parameter of the Heat model
        eval_param = model.central_param

        # calculating the kernel widths for the data based on Silverman's rule of thumb
        data_stdevs = calc_kernel_width(data)

        # evaluate the three-variate joint density
        slice = np.array([0,1,2])

        (central_param_density, all_res) = evaluate_density(param = eval_param,
                                                            model = model,
                                                            data = data,
                                                            data_transformation = DataIdentity(), # no data transformatio,
                                                            data_stdevs = data_stdevs,
                                                            slice = slice)

        # all_res is the concatenation of the evaluated parameter, the simulation result arising from that parameter and the inferred paramter density. Decompose as follows:
        eval_param = all_res[0:model.param_dim]
        sim_result = all_res[model.param_dim:model.param_dim+model.data_dim]
        central_param_density = all_res[-1]

    """

    # Build the full parameter vector for evaluation based on the passed param slice and the constant central points
    fullParam = model.central_param.copy()
    fullParam[slice] = param

    # Check if the tried parameter is within the just-defined bounds and return the lowest possible density if not.
    if (
        np.any(param < model.param_limits[slice, 0])
        or np.any(param > model.param_limits[slice, 1])
        or not model.param_is_within_domain(fullParam)
    ):
        logger.info(
            "Parameters outside of predefined range"
        )  # Slows down the sampling to much? -> Change logger level to warning or even error
        return 0.0, np.zeros(slice.shape[0] + model.data_dim + 1)

    # If the parameter is within the valid ranges...
    else:
        # Try evaluating the model for the given parameters. Evaluating the model and the jacobian for the specified parameter simultaneously provide a little speedup over calculating it separately in some cases.
        try:
            sim_res, model_jac = model.forward_and_jacobian(fullParam)
        except Exception as e:
            logger.error(
                "Error while evaluating model and its jacobian."
                "The program will continue, but the density will be set to 0."
                f"The parameter that caused the error is: {fullParam}"
                f"The error message is: {e}"
            )
            return 0, np.zeros(slice.shape[0] + model.data_dim + 1)

        # normalize sim_res
        transformed_sim_res = data_transformation.transform(sim_res)

        # Evaluate the data density in the simulation result.
        densityEvaluation = eval_kde_gauss(
            data, transformed_sim_res, data_stdevs
        )

        # Calculate the simulation model's pseudo-determinant in the parameter point (also called the correction factor).
        # Scale with the determinant of the transformation matrix.
        transformation_jac = data_transformation.jacobian(sim_res)
        correction = calc_gram_determinant(
            jnp.dot(
                transformation_jac, model_jac
            )  # We use dot, because matmul does not support scalars
        )

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
    data_transformation: DataTransformation,
    data_stdevs: np.ndarray,
    slice: np.ndarray,
) -> Tuple[np.double, np.ndarray]:
    """Calculate the logarithmical parameter density as backtransformed data density using the simulation model

    .. math::

        \\log{\\Phi_\\mathcal{Q}(q)} :=
            \\begin{cases}
                \\log{\\Phi_\\mathcal{Q}(q)} \\quad \\text{if } \\Phi_\\mathcal{Q}(q) > 0 \\\\
                -\\infty \\quad \\text{else}
            \\end{cases}

    Args:
        param (np.ndarray): parameter for which the transformed density shall be evaluated
        model (Model): model to be evaluated
        data (np.ndarray): data for the model. 2D array with shape (#num_data_points, #data_dim)
        data_transformation (DataTransformation): The data transformation used to normalize the data.
        data_stdevs (np.ndarray): array of suitable kernel width for each data dimension
        slice (np.ndarray): slice of the parameter vector that is to be evaluated

    Returns:
        Tuple[np.double, np.ndarray]:
            : natural log of the parameter density at the point param
            : sampler_results (array concatenation of parameters, simulation results and evaluated density, stored as "blob" by the emcee sampler)

    """
    trafo_density_evaluation, evaluation_results = evaluate_density(
        param, model, data, data_transformation, data_stdevs, slice
    )
    if trafo_density_evaluation == 0:
        return -np.inf, evaluation_results
    return np.log(trafo_density_evaluation), evaluation_results


def calc_gram_determinant(jac: jnp.ndarray) -> jnp.double:
    """Evaluate the pseudo-determinant of the jacobian

    .. math::

        \\sqrt{\\det \\left({\\frac{ds}{dq}(q)}^\\intercal {\\frac{ds}{dq}(q)}\\right)}

    .. warning::

        The pseudo-determinant of the model jacobian serves as a correction term in the :py:func:`evaluate_density <evaluate_density>` function.
        Therefore this function returns 0 if the result is not finite.

    Args:
      jac(jnp.ndarray): The jacobian for which the pseudo determinant shall be calculated

    Returns:
        jnp.double: The pseudo-determinant of the jacobian. Returns 0 if the result is not finite.

    Examples:

    .. code-block:: python

        import jax.numpy as jnp
        from eulerpi.core.transformations import calc_gram_determinant

        jac = jnp.array([[1,2], [3,4], [5,6], [7,8]])
        pseudo_det = calc_gram_determinant(jac)

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
