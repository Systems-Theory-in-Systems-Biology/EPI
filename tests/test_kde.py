import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from epic.core.kernel_density_estimation import evalKDECauchy, evalKDEGauss


def test_KDE1DataPoint():
    """Test both kernel density estimators by using one data point and evaluating the Kernel Density Estimation over a grid"""
    # define the one data point
    data = np.array([[0.5, 2.0]])

    # define kernel standard deviations
    stdevs = np.array([2.0, 1.0])

    # number of test grid points
    resolution = 33

    xGrid = np.linspace(-2.5, 3.5, resolution)
    yGrid = np.linspace(0.5, 3.5, resolution)

    xMesh, yMesh = np.meshgrid(xGrid, yGrid)

    gaussEvals = np.zeros((resolution, resolution))
    cauchyEvals = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            evalPoint = np.array([xMesh[i, j], yMesh[i, j]])

            gaussEvals[i, j] = evalKDEGauss(data, evalPoint, stdevs)
            cauchyEvals[i, j] = evalKDECauchy(data, evalPoint, stdevs)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xMesh,
        yMesh,
        gaussEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xMesh,
        yMesh,
        cauchyEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()


def test_KDE3DataPoints():
    # define the one data point
    data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])

    # define kernel standard deviations
    stdevs = np.array([0.5, 1.0])

    # number of test grid points
    resolution = 33

    xGrid = np.linspace(-1.0, 3.0, resolution)
    yGrid = np.linspace(-1.0, 2.0, resolution)

    xMesh, yMesh = np.meshgrid(xGrid, yGrid)

    gaussEvals = np.zeros((resolution, resolution))
    cauchyEvals = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            evalPoint = np.array([xMesh[i, j], yMesh[i, j]])

            gaussEvals[i, j] = evalKDEGauss(data, evalPoint, stdevs)
            cauchyEvals[i, j] = evalKDECauchy(data, evalPoint, stdevs)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xMesh,
        yMesh,
        gaussEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xMesh,
        yMesh,
        cauchyEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()
