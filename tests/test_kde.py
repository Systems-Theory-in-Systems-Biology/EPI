"""Tests for the Kernel Density Estimation module"""

import jax.scipy.stats as jstats
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import cm

from eulerpi.estimation import CauchyKDE, GaussKDE
from eulerpi.inference_engines.grid_inference.grids.equidistant_grid import (
    EquidistantGrid,
)


def kernel_estimators():
    """Yields the kernel density estimators"""
    yield GaussKDE
    yield CauchyKDE


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


@pytest.mark.parametrize("KDE_T", kernel_estimators())
@pytest.mark.parametrize("batch", [False, True])
def test_KDE_batch(batch, KDE_T):
    """Test both kernel density estimators by using random data points and evaluating the Kernel Density Estimation at one point

    Args:
      batch:
      KDE_T:

    Returns:

    """
    # Define random data points in 2D
    data_dim = 2
    num_data_points = 3
    data = np.random.rand(num_data_points, data_dim)
    kde = KDE_T(data)
    # define the evaluation point(s)
    n_samples = 5
    if batch:
        evalPoint = np.random.rand(n_samples, data_dim)
    else:
        evalPoint = np.array([0.5] * data_dim)

    # evaluate the KDE
    evaluated = kde(evalPoint)

    # The additional dimension should be still there if batch is True
    if batch:
        assert evaluated.shape == (n_samples,)
    else:
        assert evaluated.shape == ()


@pytest.mark.parametrize("data, stdevs, grid_bounds", kde_data_test_set())
@pytest.mark.parametrize("KDE_T", kernel_estimators())
def test_KDE_data(KDE_T, data, stdevs, grid_bounds, resolution=33):
    """Test both kernel density estimators by using one data point and evaluating the Kernel Density Estimation over a grid

    Args:
      KDE_T
      data:
      stdevs:
      grid_bounds:
      resolution:  (Default value = 33)

    Returns:

    """

    xGrid = np.linspace(*(grid_bounds[0]), resolution)
    yGrid = np.linspace(*(grid_bounds[1]), resolution)

    xMesh, yMesh = np.meshgrid(xGrid, yGrid)

    kde = KDE_T(data)

    # We only want to vectorize the call for the evaluation points in the mesh, not for the data points.
    # Map over axis 0 because the grid points are stored row-wise in the mesh
    evaluated = kde(np.stack([xMesh, yMesh], axis=-1))

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


# WARNING: The following code only works for the simplest case. Equidistant grid, same number of points in each dimension, ...
def integrate(z, x, y):
    # Integrate the function over the grid
    integral = np.trapz(np.trapz(z, y, axis=0), x, axis=0)
    return integral


@pytest.mark.parametrize("dim", [1, 2], ids=["1D", "2D"])
def test_kde_convergence_gauss(
    dim, num_grid_points=100, num_data_points=10000
):
    """Test whether the KDE converges to the true distribution."""
    # Generate random numbers from a normal distribution
    data = np.random.randn(num_data_points, dim)

    # Define the grid
    num_grid_points = np.array(
        [num_grid_points for _ in range(dim)], dtype=np.int32
    )
    limits = np.array([[-5, 5] for _ in range(dim)])
    grid = EquidistantGrid(limits, num_grid_points).grid_points
    kde = GaussKDE(data)

    kde_on_grid = kde(grid)  # Evaluate the KDE
    mean = np.zeros(dim)
    cov = np.eye(dim)
    exact_on_grid = jstats.multivariate_normal.pdf(
        grid, mean, cov
    )  # Evaluate the true distribution
    diff = np.abs(kde_on_grid - exact_on_grid)  # difference between the two

    # Plot the KDE
    import matplotlib.pyplot as plt

    if dim == 1:
        grid = grid[:, 0]
        error = np.trapz(diff, grid)  # Calculate the error
        assert error < 0.1  # ~0.06 for 100 grid points, 1000 data points

        plt.plot(grid, kde_on_grid)
        plt.plot(grid, exact_on_grid)

    elif dim == 2:
        # Calculate the error
        diff = diff.reshape(num_grid_points[0], num_grid_points[1])
        x = np.linspace(limits[0, 0], limits[0, 1], num_grid_points[0])
        y = np.linspace(limits[1, 0], limits[1, 1], num_grid_points[1])
        error = integrate(diff, x, y)
        assert error < 0.15  # ~0.13 for 100 grid points, 1000 data points
        # Surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        grid_2d = grid.reshape(num_grid_points[0], num_grid_points[1], dim)

        exact_on_grid_2d = exact_on_grid.reshape(
            num_grid_points[0], num_grid_points[1]
        )
        surf = ax.plot_surface(
            grid_2d[:, :, 0],
            grid_2d[:, :, 1],
            exact_on_grid_2d,
            alpha=0.7,
            label="exact",
        )
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d

        kde_on_grid_2d = kde_on_grid.reshape(
            num_grid_points[0], num_grid_points[1]
        )
        surf = ax.plot_surface(
            grid_2d[:, :, 0],
            grid_2d[:, :, 1],
            kde_on_grid_2d,
            alpha=0.7,
            label="kde",
        )
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d

    plt.show()
