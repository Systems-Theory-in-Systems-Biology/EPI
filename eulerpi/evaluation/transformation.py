"""This module implements the random variable transformation of the EPI algorithm.
"""

import logging
from typing import Tuple

import jax.numpy as jnp
import numpy as np

from eulerpi.data_transformations.data_transformation import DataTransformation
from eulerpi.models.base_model import BaseModel

from .gram_determinant import calc_gram_determinant
from .kde import KDE

logger = logging.getLogger(__name__)


def evaluate_density(
    param: np.ndarray,
    model: BaseModel,
    data_transformation: DataTransformation,
    kde: KDE,
    slice: np.ndarray,
) -> Tuple[np.double, np.ndarray]:
    """Calculate the parameter density as backtransformed data density using the simulation model

    .. math::

        \\Phi_\\mathcal{Q}(q) = \\Phi_\\mathcal{Y}(s(q)) \\cdot \\sqrt{\\det \\left({\\frac{ds}{dq}(q)}^\\intercal {\\frac{ds}{dq}(q)}\\right)}

    Args:
        param (np.ndarray): parameter for which the transformed density shall be evaluated
        model (BaseModel): model to be evaluated
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
        from eulerpi.evaluation.kde import GaussKDE
        from eulerpi.data_transformations import DataIdentity
        from eulerpi.transformations import evaluate_density

        # use the heat model
        model = Heat()

        # generate 1000 artificial, 5D data points for the Heat example model
        data_mean = np.array([0.5, 0.1, 0.5, 0.9, 0.5])
        data = np.random.randn(1000, 5)/25.0 + data_mean

        # evaluating the parameter probabiltiy density at the central parameter of the Heat model
        param = model.central_param

        # Create a kernel density estimation using the gaussian kernels with widths based on Silverman's rule of thumb
        kde = GaussKDE(data)

        # evaluate the three-variate joint density
        slice = np.array([0,1,2])

        # The evaluate_density function returns the parameter itself, the simulation result, and the inferred density of the parameter
        param, sim_res, param_density = evaluate_density(param = param,
                                                            model = model,
                                                            data_transformation = DataIdentity(), # no data transformation,
                                                            kde = kde,
                                                            slice = slice)

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
        return np.zeros((slice.shape[0],)), np.zeros((model.data_dim,)), 0.0

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
            return (
                np.zeros((slice.shape[0],)),
                np.zeros((model.data_dim,)),
                0.0,
            )

        # normalize sim_res
        transformed_sim_res = data_transformation.transform(sim_res)

        # Evaluate the data density in the simulation result.
        densityEvaluation = kde(transformed_sim_res)

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

        return param, sim_res, trafo_density_evaluation


def evaluate_log_density(
    param: np.ndarray,
    model: BaseModel,
    data_transformation: DataTransformation,
    kde: KDE,
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
        model (BaseModel): model to be evaluated
        data_transformation (DataTransformation): The data transformation used to normalize the data.
        data_stdevs (np.ndarray): array of suitable kernel width for each data dimension
        slice (np.ndarray): slice of the parameter vector that is to be evaluated

    Returns:
        Tuple[np.double, np.ndarray]:
            : natural log of the parameter density at the point param
            : sampler_results (array concatenation of parameters, simulation results and evaluated density, stored as "blob" by the emcee sampler)

    """
    param, sim_res, density = evaluate_density(
        param, model, data_transformation, kde, slice
    )

    if density == 0:
        return param, sim_res, -np.inf
    return param, sim_res, np.log(density)
