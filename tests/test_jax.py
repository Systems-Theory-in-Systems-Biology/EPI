import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_jax_cpu():
    jax.default_device = jax.devices("cpu")[0]
    do_matrix_multiplication()


# TODO: Find better approach to check if gpu is available and accessible
# Jax uses gpu per default if available. So when can check wether gpu is available by checking for the default
@pytest.mark.xfail(
    jax.default_backend() == "cpu",
    reason="GPU not available, maybe jax[cuda] or cuda + cudnn not installed",
)
def test_jax_gpu():
    jax.default_device = jax.devices("gpu")[0]
    do_matrix_multiplication()


def do_matrix_multiplication(n: int = 5000):
    x = np.random.randn(n, n)
    y = np.random.randn(n, n)

    xj = jnp.array(x)
    yj = jnp.array(y)

    zj = jnp.dot(xj, yj)
