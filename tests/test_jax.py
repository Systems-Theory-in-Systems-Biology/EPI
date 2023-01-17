import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_jax_cpu():
    """Test wether jax can run on the cpu by executing some jax code"""
    jax.default_device = jax.devices("cpu")[0]
    do_matrix_multiplication()


# The jax gpu test may fail if no nvidia gpu is available or the cuda and cudnn libraries are not installed
@pytest.mark.xfail(
    jax.default_backend()
    == "cpu",  # Jax uses gpu per default if available. So when can check wether gpu is available by checking for the default
    reason="GPU not available, maybe jax[cuda] or cuda + cudnn not installed",
)
def test_jax_gpu():
    """Test wether jax can run on the gpu by executing some jax code"""
    jax.default_device = jax.devices("gpu")[0]
    do_matrix_multiplication()


def do_matrix_multiplication(n: int = 5000):
    """Do a simple matrix-matrix multiplication

    :param n: number of entries per dimension, defaults to 5000
    :type n: int, optional
    """
    x = np.random.randn(n, n)
    y = np.random.randn(n, n)

    xj = jnp.array(x)
    yj = jnp.array(y)

    zj = jnp.dot(xj, yj)
