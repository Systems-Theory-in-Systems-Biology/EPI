"""
Contains grid based evaluations and plotting of models, which implement simple transformations: Linear, Exponential, LinearODE
"""

import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from matplotlib import cm

from epi.core.inference import InferenceType, inference
from epi.core.kde import calc_kernel_width, eval_kde_gauss
from epi.core.transformations import evaluate_density
from epi.examples.simple_models import Exponential, Linear, LinearODE


def test_transformationLinear():
    """ """
    model = Linear()

    # create approx. 1000 data points that are perfectly uniformly distributed over a grid
    # the range of data points is the 2D interval [0,10]x[-2,-4]
    dataResolution = 33

    xGrid = np.linspace(0.0, 10.0, dataResolution)
    yGrid = np.linspace(-4.0, -2.0, dataResolution)
    xMesh, yMesh = np.meshgrid(xGrid, yGrid)

    data = np.array([xMesh.flatten(), yMesh.flatten()]).T

    # define standard deviations according to silverman
    data_stdevs = calc_kernel_width(data)

    # Now plot the data Gaussian KDE
    KDEresolution = 25

    # the KDE Grid is 40% larger than the interval of the data and has a different resolution
    KDExGrid = np.linspace(-2.0, 12.0, KDEresolution)
    KDEyGrid = np.linspace(-4.4, -1.6, KDEresolution)
    KDExMesh, KDEyMesh = np.meshgrid(KDExGrid, KDEyGrid)

    gaussEvals = np.zeros((KDEresolution, KDEresolution))

    for i in range(KDEresolution):
        for j in range(KDEresolution):
            evalPoint = np.array([KDExMesh[i, j], KDEyMesh[i, j]])
            gaussEvals[i, j] = eval_kde_gauss(data, evalPoint, data_stdevs)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("Data KDE")
    surf = ax.plot_surface(
        KDExMesh,
        KDEyMesh,
        gaussEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()

    paramResolution = 15
    paramxGrid = np.linspace(-0.2, 1.2, paramResolution)
    paramyGrid = np.linspace(-0.2, 1.2, paramResolution)

    paramxMesh, paramyMesh = np.meshgrid(paramxGrid, paramyGrid)

    paramEvals = np.zeros((paramResolution, paramResolution))

    # TODO: Use dense grid evaluation function, but pay attention to the changed limits above. Then todo below can also be removed
    # TODO: Evaluate Density is missing the slice argument here
    for i in range(paramResolution):
        for j in range(paramResolution):
            paramPoint = np.array([paramxMesh[i, j], paramyMesh[i, j]])
            paramEvals[i, j], _ = evaluate_density(
                paramPoint,
                model,
                data,
                data_stdevs,
                slice=np.arange(model.param_dim),
            )

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("Parameter Density Estimation")
    surf = ax.plot_surface(
        paramxMesh,
        paramyMesh,
        paramEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()


def test_transformationExponential():
    """ """
    model = Exponential()

    # create true parameter points that are drawn uniformly from [0,1]^2

    num_data_points = 10000
    trueParam = np.random.rand(num_data_points, 2) + 1

    data = vmap(model.forward, in_axes=0)(trueParam)

    # define standard deviations according to silverman
    data_stdevs = calc_kernel_width(data)

    # Now plot the data Gaussian KDE
    KDEresolution = 25

    KDExGrid = np.linspace(0.8 * np.exp(1), 2.2 * np.exp(1), KDEresolution)
    KDEyGrid = np.linspace(np.exp(0.8), np.exp(2.2), KDEresolution)
    KDExMesh, KDEyMesh = np.meshgrid(KDExGrid, KDEyGrid)

    gaussEvals = np.zeros((KDEresolution, KDEresolution))

    for i in range(KDEresolution):
        for j in range(KDEresolution):
            evalPoint = np.array([KDExMesh[i, j], KDEyMesh[i, j]])
            gaussEvals[i, j] = eval_kde_gauss(data, evalPoint, data_stdevs)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("Data KDE")
    surf = ax.plot_surface(
        KDExMesh,
        KDEyMesh,
        gaussEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()

    paramResolution = 25
    paramxGrid = np.linspace(0.8, 2.2, paramResolution)
    paramyGrid = np.linspace(0.8, 2.2, paramResolution)

    paramxMesh, paramyMesh = np.meshgrid(paramxGrid, paramyGrid)

    paramEvals = np.zeros((paramResolution, paramResolution))
    fullSlice = np.arange(model.param_dim)

    for i in range(paramResolution):
        for j in range(paramResolution):
            paramPoint = np.array([paramxMesh[i, j], paramyMesh[i, j]])
            paramEvals[i, j], _ = evaluate_density(
                paramPoint, model, data, data_stdevs, slice=fullSlice
            )

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("Parameter Density Estimation")
    surf = ax.plot_surface(
        paramxMesh,
        paramyMesh,
        paramEvals,
        alpha=0.75,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()


def test_transformationODELinear():
    """Exemplary application of Eulerian Parameter Inference a very simplistic ordinary differential equation model.
    We create our toy data by first defining a true parameter distribution.
    The actual data is then obtained by evaluating the model in parameters drawn from this true distribution.
    Ideally, we would be able to reconstruct the true parameter density.
    However, we will see that this parameter inference problem possesses more than one solution and is therefore not well-posed.

    Args:

    Returns:

    """

    # define the model
    model = LinearODE()

    # generate artificial data
    num_data_points = 10000
    params = model.generate_artificial_params(num_data_points)
    data = model.generate_artificial_data(params)

    # run MCMC sampling for EPI
    inference(model, data, InferenceType.MCMC)
