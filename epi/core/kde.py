"""This module provides functions to handle the Kernel Densitiy Estimation (KDE_) in EPI.


.. _KDE: https://en.wikipedia.org/wiki/Kernel_density_estimation
"""

import typing

import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import cauchy, norm


@jit
def eval_kde_cauchy(
    data: jnp.ndarray, simRes: jnp.ndarray, scales: jnp.ndarray
) -> typing.Union[jnp.double, jnp.ndarray]:
    r"""Evaluates a Cauchy Kernel Density estimator in one or several simulation results.
        Assumes that each data point is a potentially high-dimensional sample
        from a joint data distribution.
        This is for example given for time-series data, where each evaluation
        time is one dimension of the data point.
        In the following formula x are the evaluation points (simRes) and y is the data.

        .. math::
            density_{i} = \frac{1}{samples} \sum_{s=1}^{samples} \prod_{d=1}^{dims} \frac{1}{(\frac{x_{i,d} - y_{s,d}}{scales_d})^2 \; \pi \; scales_d}

    Args:
      data(jnp.ndarray): data for the model: 2D array with shape (#Samples, #MeasurementDimensions)
      simRes(jnp.ndarray): evaluation coordinates array of shape (#nEvals, #MeasurementDimensions) or (#MeasurementDimensions,)
      scales(jnp.ndarray): one scale for each dimension

    Returns:
        typing.Union[jnp.double, jnp.ndarray]: estimated kernel density evaluated at the simulation result(s), shape: (#nEvals,) or ()

    """

    return (
        jnp.sum(
            jnp.prod(
                cauchy.pdf(simRes[..., jnp.newaxis, :], data, scales),
                axis=-1,  # prod over #measurementDimensions
            ),
            axis=-1,  # sum over sampleDim
        )
        / data.shape[0]
    )


@jit
def eval_kde_gauss(
    data: jnp.ndarray, simRes: jnp.ndarray, scales: jnp.ndarray
) -> typing.Union[jnp.double, jnp.ndarray]:
    """Evaluates a Gaussian Kernel Density estimator in one or severalsimulation result.
    Assumes that each data point is a potentially high-dimensional sample from a joint data distribution.
    This is for example given for time-series data, where each evaluation time is one dimension of the data point.
    While it is possible to define different standard deviations for different measurement dimensions, it is so far not possible to define covariances.

    Args:
      data(jnp.ndarray): data for the model: 2D array with shape (#Samples, #MeasurementDimensions)
      simRes(jnp.ndarray): evaluation coordinates array of shape (#nEvals, #MeasurementDimensions) or (#MeasurementDimensions,)
      scales(jnp.ndarray): one scale for each dimension

    Returns:
        typing.Union[jnp.double, jnp.ndarray]: estimated kernel density evaluated at the simulation result(s), shape: (#nEvals,) or ()
    """

    return (
        jnp.sum(
            jnp.prod(
                norm.pdf(simRes[..., jnp.newaxis, :], data, scales),
                axis=-1,  # prod over #measurementDimensions
            ),
            axis=-1,  # sum over sampleDim
        )
        / data.shape[0]
    )


@jit
def calc_kernel_width(data: jnp.ndarray) -> jnp.ndarray:
    """Sets the width of the kernels used for density estimation of the data according to the Silverman rule

    Args:
        data(jnp.ndarray): data for the model: 2D array with shape (#Samples, #MeasurementDimensions)

    Returns:
        jnp.ndarray: kernel width for each data dimension, shape: (#MeasurementDimensions,)

    """
    num_data_points, data_dim = data.shape
    stdevs = jnp.std(data, axis=0, ddof=1)

    # Silvermans rule
    return stdevs * (num_data_points * (data_dim + 2) / 4.0) ** (
        -1.0 / (data_dim + 4)
    )
