"""This module provides functions to handle the Kernel Densitiy Estimation (KDE_) in EPI.

    It is used in the EPI algorithm to :py:func:`eulerpi.transformations.evaluate_density <evaluate the density>` of the transformed data distribution at the simulation results.


.. _KDE: https://en.wikipedia.org/wiki/Kernel_density_estimation
"""

import jax.numpy as jnp
from jax import jit


@jit
def calc_silverman_kernel_width(data: jnp.ndarray) -> jnp.ndarray:
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
        from eulerpi.estimation.kde import calc_kernel_width

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
