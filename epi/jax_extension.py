from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp


def value_and_jacfwd(fun: Callable[[jnp.ndarray], jnp.ndarray]):
    """Returns a function that computes the value and the jacobian of the passed function using forward mode AD.

    Args:
      fun(Callable[[jnp.ndarray], jnp.ndarray]): The function to supplement with the jacobian
      fun: Callable[[jnp.ndarray]:
      jnp.ndarray]:

    Returns:

    """

    def value_and_jacfwd_fun(x: jnp.ndarray):
        """

        Args:
          x: jnp.ndarray:

        Returns:

        """
        pushfwd = partial(jax.jvp, fun, (x,))
        y, jac = jax.vmap(pushfwd, out_axes=(None, -1))(
            (jnp.eye(x.shape[0], dtype=x.dtype),)
        )
        return y, jac

    return value_and_jacfwd_fun


def value_and_jacrev(fun: Callable[..., jnp.ndarray]):
    """Returns a function that computes the value and the jacobian of the passed function using reverse mode AD.

    Args:
      fun(Callable[..., jnp.ndarray]): The function to supplement with the jacobian
      fun: Callable[...:
      jnp.ndarray]:

    Returns:

    """

    def value_and_jacrev_fun(x):
        """

        Args:
          x:

        Returns:

        """
        y, pullback = jax.vjp(fun, x)
        jac = jax.vmap(pullback)(jnp.eye(y.shape[0], dtype=y.dtype))[0]
        return y, jac

    return value_and_jacrev_fun
