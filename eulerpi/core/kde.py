"""This module provides functions to handle the Kernel Densitiy Estimation (KDE_) in EPI.

    It is used in the EPI algorithm to :py:func:`eulerpi.core.transformations.evaluate_density <evaluate the density>` of the transformed data distribution at the simulation results.


.. _KDE: https://en.wikipedia.org/wiki/Kernel_density_estimation
"""

import typing

import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import cauchy, norm


@jit
def eval_kde_cauchy(
    data: jnp.ndarray, sim_res: jnp.ndarray, scales: jnp.ndarray
) -> typing.Union[jnp.double, jnp.ndarray]:
    r"""
    Evaluates a Cauchy Kernel Density estimator in one or several simulation results.
    Assumes that each data point is a potentially high-dimensional sample from a joint data distribution.
    This is for example given for time-series data, where each evaluation time is one dimension of the data point.
    In the following formula x are the evaluation points (sim_res) and y is the data.

        .. math::
            density_{i} = \frac{1}{samples} \sum_{s=1}^{samples} \prod_{d=1}^{dims} \frac{1}{(\frac{x_{i,d} - y_{s,d}}{scales_d})^2 \; \pi \; scales_d}

    Args:
      data(jnp.ndarray): data for the model: 2D array with shape (#Samples, #MeasurementDimensions)
      sim_res(jnp.ndarray): evaluation coordinates array of shape (#nEvals, #MeasurementDimensions) or (#MeasurementDimensions,)
      scales(jnp.ndarray): one scale for each dimension

    Returns:
        typing.Union[jnp.double, jnp.ndarray]: estimated kernel density evaluated at the simulation result(s), shape: (#nEvals,) or ()

    """

    return (
        jnp.sum(
            jnp.prod(
                cauchy.pdf(sim_res[..., jnp.newaxis, :], data, scales),
                axis=-1,  # prod over #measurementDimensions
            ),
            axis=-1,  # sum over sampleDim
        )
        / data.shape[0]
    )


@jit
def eval_kde_gauss(
    data: jnp.ndarray, sim_res: jnp.ndarray, scales: jnp.ndarray
) -> typing.Union[jnp.double, jnp.ndarray]:
    """Evaluates a Gaussian Kernel Density estimator in one or severalsimulation result.
    Assumes that each data point is a potentially high-dimensional sample from a joint data distribution.
    This is for example given for time-series data, where each evaluation time is one dimension of the data point.
    While it is possible to define different standard deviations for different measurement dimensions, it is so far not possible to define covariances.

    Args:
        data(jnp.ndarray): data for the model: 2D array with shape (#Samples, #MeasurementDimensions)
        sim_res(jnp.ndarray): evaluation coordinates array of shape (#nEvals, #MeasurementDimensions) or (#MeasurementDimensions,)
        scales(jnp.ndarray): one scale for each dimension

    Returns:
        typing.Union[jnp.double, jnp.ndarray]: estimated kernel density evaluated at the simulation result(s), shape: (#nEvals,) or ()

    .. note::

        Make sure to always use 2D arrays as data, especially when the data dimension is only one.\n
        The data object should be shaped (#Samples, 1) and not (#Samples,) in this case.

    Examples:

    .. code-block:: python

        import jax.numpy as jnp
        from eulerpi.core.kde import eval_kde_gauss

        # create 4 data points of dimension 2 and store them in a numpy 2D array
        data = jnp.array([[0,0], [0,1], [1,0], [1,1]])

        # we intend to evaluate the kernel density estimator at the point (0.5, 0.5)
        evaluation_coordinates = jnp.array([[0.5, 0.5]])

        # the dimension-specific kernel bandwidths are set to 1
        scales = jnp.array([1,1])

        kde_res = eval_kde_gauss(data, evaluation_coordinates, scales)

    """

    return (
        jnp.sum(
            jnp.prod(
                norm.pdf(sim_res[..., jnp.newaxis, :], data, scales),
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

    .. note::

        Make sure to always use 2D arrays as data, especially when the data dimension is only one.\n
        The data object should be shaped (#Samples, 1) and not (#Samples,) in this case.

    Examples:

    .. code-block:: python

        import jax.numpy as jnp
        from eulerpi.core.kde import calc_kernel_width

        # create 4 data points of dimension 2 and store them in a numpy 2D array
        data = jnp.array([[0,0], [0,2], [1,0], [1,2]])

        scales = calc_kernel_width(data)

    """
    num_data_points, data_dim = data.shape
    stdevs = jnp.std(data, axis=0, ddof=1)

    # Silvermans rule
    return stdevs * (num_data_points * (data_dim + 2) / 4.0) ** (
        -1.0 / (data_dim + 4)
    )
