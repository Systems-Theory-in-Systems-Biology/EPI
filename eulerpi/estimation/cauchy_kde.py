import typing

import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import cauchy

from .kde import KDE


class CauchyKDE(KDE):
    def __call__(self, data_point):
        return eval_kde_cauchy(self.data, data_point, self.kernel_width)


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
