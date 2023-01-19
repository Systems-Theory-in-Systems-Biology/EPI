import jax
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import cm

from epic.core.kernel_density_estimation import (
    calcKernelWidth,
    evalKDECauchy,
    evalKDEGauss,
)


def kernel_estimators():
    """Yields the kernel density estimators"""
    yield evalKDECauchy
    yield evalKDEGauss


def kde_data_test_set():
    """Yields test data for the KDE data tests"""
    data_list = [
        np.array([[0.5, 2.0]]),
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]),
    ]
    stdev_list = [np.array([2.0, 1.0]), np.array([0.5, 1.0])]
    grid_bounds_list = [[[-2.5, 3.5], [0.5, 3.5]], [[-1.0, 3.0], [-1, 2.0]]]
    for i in range(len(data_list)):
        yield data_list[i], stdev_list[i], grid_bounds_list[i]


@pytest.mark.parametrize("evalKDE", kernel_estimators())
@pytest.mark.parametrize("batch", [False, True])
def test_KDE_batch(batch, evalKDE):
    """Test both kernel density estimators by using random data points and evaluating the Kernel Density Estimation at one point"""
    # Define random data points in 2D
    dataDim = 2
    numDataPoints = 3
    data = np.random.rand(numDataPoints, dataDim)
    stdevs = calcKernelWidth(data)

    # define the evaluation point(s)
    n_samples = 5
    if batch:
        evalPoint = np.random.rand(n_samples, dataDim)
    else:
        evalPoint = np.array([0.5] * dataDim)

    # evaluate the KDE
    evaluated = evalKDE(data, evalPoint, stdevs)

    # The additional dimension should be still there if batch is True
    if batch:
        assert evaluated.shape == (n_samples,)
    else:
        assert evaluated.shape == ()


@pytest.mark.parametrize("data, stdevs, grid_bounds", kde_data_test_set())
@pytest.mark.parametrize("evalKDE", kernel_estimators())
def test_KDE_data(evalKDE, data, stdevs, grid_bounds, resolution=33):
    """Test both kernel density estimators by using one data point and evaluating the Kernel Density Estimation over a grid"""

    xGrid = np.linspace(*(grid_bounds[0]), resolution)
    yGrid = np.linspace(*(grid_bounds[1]), resolution)

    xMesh, yMesh = np.meshgrid(xGrid, yGrid)

    # We only want to vectorize the call for the evaluation points in the mesh, not for the data points.
    # Map over axis 0 because the grid points are stored row-wise in the mesh
    evaluated = jax.vmap(evalKDE, in_axes=(None, 0, None))(
        data, np.stack([xMesh, yMesh], axis=-1), stdevs
    )
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xMesh,
        yMesh,
        evaluated,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()
