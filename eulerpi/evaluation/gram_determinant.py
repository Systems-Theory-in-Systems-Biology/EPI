import jax.numpy as jnp
from jax import jit

from eulerpi.logger import logger


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
        from eulerpi.transformations import calc_gram_determinant

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
