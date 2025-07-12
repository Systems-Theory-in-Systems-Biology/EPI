from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp


def value_and_jacfwd(
    fun: Callable[[jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """Returns a function that computes the value and the jacobian of the passed function using forward mode AD.

    Args:
      fun(Callable[[jnp.ndarray], jnp.ndarray]): The function to supplement with the jacobian

    Returns:
      typing.Callable[[jnp.ndarray], typing.Tuple[jnp.ndarray, jnp.ndarray]]: A function that computes the value and the jacobian of the passed function using forward mode AD.

    """

    def value_and_jacfwd_fun(
        x: jnp.ndarray,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """

        Args:
          x(jnp.ndarray): The input to the function

        Returns:
          typing.Tuple[jnp.ndarray, jnp.ndarray]: The value and the jacobian of the passed function using forward mode AD.

        """
        pushfwd = partial(jax.jvp, fun, (x,))
        y, jac = jax.vmap(pushfwd, out_axes=(None, -1))(
            (jnp.eye(x.shape[0], dtype=x.dtype),)
        )
        return y, jac

    return value_and_jacfwd_fun


def value_and_jacrev(
    fun: Callable[..., jnp.ndarray],
) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """Returns a function that computes the value and the jacobian of the passed function using reverse mode AD.

    Args:
      fun(Callable[..., jnp.ndarray]): The function to supplement with the jacobian

    Returns:
      typing.Callable[[jnp.ndarray], typing.Tuple[jnp.ndarray, jnp.ndarray]]: A function that computes the value and the jacobian of the passed function using reverse mode AD.

    """

    def value_and_jacrev_fun(x):
        """

        Args:
          x(jnp.ndarray): The input to the function

        Returns:
          typing.Tuple[jnp.ndarray, jnp.ndarray]: The value and the jacobian of the passed function using reverse mode AD.

        """
        y, pullback = jax.vjp(fun, x)
        jac = jax.vmap(pullback)(jnp.eye(y.shape[0], dtype=y.dtype))[0]
        return y, jac

    return value_and_jacrev_fun
