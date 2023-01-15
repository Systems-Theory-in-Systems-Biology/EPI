import jax
import jax.numpy as jnp
import numpy as np
import pytest


def Devices():
    for device in ["cpu", "gpu"]:
        yield device


@pytest.mark.parametrize("device", Devices())
def test_jax(device: str):
    jax.default_device = jax.devices(device)[0]
    n = 5000

    x = np.random.randn(n, n)
    y = np.random.randn(n, n)

    xj = jnp.array(x)
    yj = jnp.array(y)

    zj = jnp.dot(xj, yj)
