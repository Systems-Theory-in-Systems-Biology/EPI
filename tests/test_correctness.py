import jax.scipy.stats as jstats
import matplotlib.pyplot as plt
import numpy as np
import pytest

from epi.core.dense_grid import generate_regular_grid
from epi.core.inference import InferenceType, inference
from epi.core.kde import calc_kernel_width, eval_kde_gauss
from epi.core.result_manager import ResultManager
from epi.examples.simple_models import LinearODE


# WARNING: The following code only works for the simplest case. Equidistant grid, same number of points in each dimension, ...
def integrate(z, x, y):
    # Integrate the function over the grid
    integral = np.trapz(np.trapz(z, y, axis=0), x, axis=0)
    return integral


@pytest.mark.parametrize("dim", [1, 2], ids=["1D", "2D"])
def test_kde_converges_gauss(dim, num_grid_points=100, num_data_points=1000):
    """Test whether the KDE converges to the true distribution."""
    # Generate random numbers from a normal distribution
    data = np.random.randn(num_data_points, dim)
    stdevs = calc_kernel_width(data)

    # Define the grid
    num_grid_points = np.array(
        [num_grid_points for _ in range(dim)], dtype=np.int32
    )
    limits = np.array([[-5, 5] for _ in range(dim)])
    grid = generate_regular_grid(num_grid_points, limits, flatten=True)

    kde_on_grid = eval_kde_gauss(data, grid, stdevs)  # Evaluate the KDE
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
        assert error < 0.1  # ~0.07

        plt.plot(grid, kde_on_grid)
        plt.plot(grid, exact_on_grid)

    elif dim == 2:
        # Calculate the error
        diff = diff.reshape(num_grid_points[0], num_grid_points[1])
        x_unique = np.unique(grid[:, 0])
        y_unique = np.unique(grid[:, 1])
        error = integrate(diff, x_unique, y_unique)
        assert error < 0.02  # ~0.0015
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


# TODO: Generalize, currently only works for dense vs mcmc
def test_linear_ode_cor():
    # define the model
    model = LinearODE()

    # generate artificial data
    num_data_points = 1000
    num_steps = 1000
    num_grid_points = 50
    params = model.generate_artificial_params(num_data_points)
    data = model.generate_artificial_data(params)

    # run EPI with all inference types
    results = {}
    full_slice = np.arange(model.param_dim)
    for inference_type in InferenceType._member_map_.values():
        result_manager = ResultManager(model.name, str(inference_type))
        if InferenceType(inference_type) == InferenceType.MCMC:
            inference(
                model,
                data,
                inference_type,
                result_manager=result_manager,
                num_steps=num_steps,
            )
        elif InferenceType(inference_type) == InferenceType.DENSE_GRID:
            inference(
                model,
                data,
                inference_type,
                result_manager=result_manager,
                num_grid_points=num_grid_points,
            )
        else:
            # next iteration
            continue
        results[inference_type] = result_manager.load_sim_results(
            full_slice, 0, 1
        )

    # define true pdf
    def true_pdf(grid, distribution="uniform"):
        if distribution == "uniform":
            val = (
                1.0
                / (
                    LinearODE.TRUE_PARAM_LIMITS[0, 1]
                    - LinearODE.TRUE_PARAM_LIMITS[0, 0]
                )
                / (
                    LinearODE.TRUE_PARAM_LIMITS[1, 1]
                    - LinearODE.TRUE_PARAM_LIMITS[1, 0]
                )
            )
            # return val where grid is in the limits and 0 otherwise
            return np.where(
                np.logical_and(
                    np.logical_and(
                        grid[:, 0] >= LinearODE.TRUE_PARAM_LIMITS[0, 0],
                        grid[:, 0] <= LinearODE.TRUE_PARAM_LIMITS[0, 1],
                    ),
                    np.logical_and(
                        grid[:, 1] >= LinearODE.TRUE_PARAM_LIMITS[1, 0],
                        grid[:, 1] <= LinearODE.TRUE_PARAM_LIMITS[1, 1],
                    ),
                ),
                val,
                0,
            )
        elif distribution == "gaussian":
            mean = np.zeros(model.param_dim) + 1.5
            cov = np.eye(model.param_dim)
            return jstats.multivariate_normal.pdf(grid, mean, cov)

    # Extract and process the results
    lims = LinearODE.PARAM_LIMITS
    x = np.linspace(lims[0, 0], lims[0, 1], num_grid_points)
    y = np.linspace(lims[1, 0], lims[1, 1], num_grid_points)
    grid = generate_regular_grid(
        np.array([num_grid_points, num_grid_points]), lims, flatten=True
    )
    grid_2d = grid.reshape(num_grid_points, num_grid_points, model.param_dim)
    # grid = results[InferenceType.DENSE_GRID][2]

    mcmc_params = results[InferenceType.MCMC][2]
    mcmc_params_density = results[InferenceType.MCMC][0]
    mcmc_kde = eval_kde_gauss(
        mcmc_params, grid, calc_kernel_width(mcmc_params)
    )

    dense_grid_pdf = results[InferenceType.DENSE_GRID][0]

    true_pdf_grid = true_pdf(grid)
    true_kde = eval_kde_gauss(params, grid, calc_kernel_width(params))
    true_pdf_samples = true_pdf(params)

    def to2d(grid):
        return grid.reshape(num_grid_points, num_grid_points)

    # DEBUGGING
    print("kernel width", calc_kernel_width(mcmc_params))
    print("true kernel width", calc_kernel_width(params))
    print("integral of mcmc kde", integrate(to2d(mcmc_kde), x, y))
    print("integral of dense grid pdf ", integrate(to2d(dense_grid_pdf), x, y))
    print("integral of true pdf ", integrate(to2d(true_pdf_grid), x, y))

    scatter_mcmc_params = False
    surf_mcmc_kde = True

    scatter_true_params = False
    surf_true_pdf_grid = True
    surf_true_kde = False

    surf_dense_grid_pdf = True

    # Surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if scatter_mcmc_params:
        ax.scatter(
            mcmc_params[:, 0], mcmc_params[:, 1], 0, label="mcmc params"
        )  # We dont know the normalizing constant, so we cant plot the density
    if scatter_true_params:
        ax.scatter(
            params[:, 0], params[:, 1], true_pdf_samples, label="true params"
        )

    if surf_mcmc_kde:
        surf = ax.plot_surface(
            grid_2d[:, :, 0],
            grid_2d[:, :, 1],
            to2d(mcmc_kde),
            alpha=0.3,
            label="mcmc kde",
        )
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d
    if surf_true_pdf_grid:
        surf = ax.plot_surface(
            grid_2d[:, :, 0],
            grid_2d[:, :, 1],
            to2d(true_pdf_grid),
            alpha=0.3,
            label="true pdf",
        )
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d
    if surf_true_kde:
        surf = ax.plot_surface(
            grid_2d[:, :, 0],
            grid_2d[:, :, 1],
            to2d(true_kde),
            alpha=0.1,
            label="true kde",
        )
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d
    if surf_dense_grid_pdf:
        surf = ax.plot_surface(
            grid_2d[:, :, 0],
            grid_2d[:, :, 1],
            to2d(dense_grid_pdf),
            alpha=0.3,
            label="dense grid pdf",
        )
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d

    ax.set_xlabel("param 1")
    ax.set_ylabel("param 2")
    ax.set_zlabel("density")

    ax.legend()
    plt.show()


if __name__ == "__main__":
    from epi import logger

    logger.setLevel("DEBUG")

    test_linear_ode_cor()
