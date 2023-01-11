import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import cauchy, norm


@jit
def evalKDECauchy(
    data: jnp.ndarray, simRes: jnp.ndarray, scales: jnp.ndarray
) -> jnp.double:
    r"""Evaluates a Cauchy Kernel Density estimator in one simulation result.
        Assumes that each data point is a potentially high-dimensional sample
        from a joint data distribution.
        This is for example given for time-series data, where each evaluation
        time is one dimension of the data point.

        .. math::
            density = \frac{1}{samples} \sum_{s=1}^{samples} \prod_{d=1}^{dims} \frac{1}{(\frac{x_d - y_{s,d}}{scales_d})^2 \; \pi \; scales_d}

    Ijnput: data (data for the model: 2D array with shape (#Samples, #MeasurementDimensions))
           simRes (evaluation coordinates array with one entry for each data dimension)
           scales (one scale for each dimension)

    :return: densityEvaluation (estimated kernel density evaluated at the simulation result)
    """
    return (
        jnp.sum(jnp.prod(cauchy.pdf(data, simRes, scales), axis=1))
        / data.shape[0]
    )


@jit
def evalKDEGauss(
    data: jnp.ndarray, simRes: jnp.ndarray, scales: jnp.ndarray
) -> jnp.double:
    """Evaluates a Gaussian Kernel Density estimator in one simulation result.
        Assumes that each data point is a potentially high-dimensional sample from a joint data distribution.
        This is for example given for time-series data, where each evaluation time is one dimension of the data point.
        While it is possible to define different standard deviations for different measurement dimensions, it is so far not possible to define covariances.

    Ijnput: data (data for the model: 2D array with shape (#Samples, #MeasurementDimensions))
           simRes (evaluation coordinates array with one entry for each data dimension)
           stdevs (one standard deviation for each dimension)

    Output: densityEvaluation (estimated kernel density evaluated at the simulation result)
    """
    return (
        jnp.sum(jnp.prod(norm.pdf(data, simRes, scales), axis=1))
        / data.shape[0]
    )


@jit
def calcKernelWidth(data: jnp.ndarray) -> jnp.double:
    """Sets the width of the kernels used for density estimation of the data according to the Silverman rule

    Input: data: 2d array with shape (#Samples, #MeasurementDimensions): data for the model

    Output: stdevs: array with shape (#MeasurementDimensions): suitable kernel standard deviations for each measurement dimension
    """
    numDataPoints, dataDim = data.shape
    stdevs = jnp.std(data, axis=0, ddof=1)

    # Silvermans rule
    return stdevs * (numDataPoints * (dataDim + 2) / 4.0) ** (
        -1.0 / (dataDim + 4)
    )
