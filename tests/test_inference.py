import jax.scipy.stats as jstats
import matplotlib.pyplot as plt
import numpy as np

from epipy.core.dense_grid import generate_regular_grid
from epipy.core.inference import InferenceType, inference
from epipy.core.kde import calc_kernel_width, eval_kde_gauss
from epipy.core.result_manager import ResultManager
from epipy.examples.simple_models import LinearODE


# WARNING: The following code only works for the simplest case. Equidistant grid, same number of points in each dimension, ...
def integrate(z, x, y):
    # Integrate the function over the grid
    integral = np.trapz(np.trapz(z, y, axis=0), x, axis=0)
    return integral


# TODO: Generalize, currently only works for dense vs mcmc
def test_inference_mcmc_dense_exact(
    num_data_points=1000,
    num_steps=3000,
    num_grid_points=50,
):
    # define the model
    model = LinearODE()

    # generate artificial data
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
            # Take every second sample and skip the first 5% of the chain
            results[inference_type] = result_manager.load_sim_results(
                full_slice, num_steps // 20, 2
            )
        elif InferenceType(inference_type) == InferenceType.DENSE_GRID:
            inference(
                model,
                data,
                inference_type,
                result_manager=result_manager,
                num_grid_points=num_grid_points,
            )
            results[inference_type] = result_manager.load_sim_results(
                full_slice, 0, 1
            )
        else:
            # skip other inference types
            continue

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

    integral_mcmc_kde = integrate(to2d(mcmc_kde), x, y)
    integral_dense_grid_pdf = integrate(to2d(dense_grid_pdf), x, y)
    integral_true_pdf = integrate(to2d(true_pdf_grid), x, y)

    # DEBUGGING
    print("integral of mcmc kde", integral_mcmc_kde)
    print("integral of dense grid pdf ", integral_dense_grid_pdf)
    print("integral of true pdf ", integral_true_pdf)

    # Just a quick check if the integrals are correct and that the range chosen limits are large enough
    threshold = 0.9  # We want to capture at least 90% of the probability mass
    # TODO: The threshold should be adapted depending on how hard the problem is
    # and how many samples / grid points we have
    assert integral_mcmc_kde > threshold
    assert integral_dense_grid_pdf > threshold
    assert integral_true_pdf > threshold

    # Calculate the errors on the grid
    mcmc_kde_error = np.abs(mcmc_kde - true_pdf_grid)
    dense_grid_pdf_error = np.abs(dense_grid_pdf - true_pdf_grid)

    # Calculate the integrals of the errors
    integral_mcmc_kde_error = integrate(to2d(mcmc_kde_error), x, y)
    integral_dense_grid_pdf_error = integrate(to2d(dense_grid_pdf_error), x, y)

    # Divide the integral through the area of the grid
    integral_mcmc_kde_error /= (lims[0, 1] - lims[0, 0]) * (
        lims[1, 1] - lims[1, 0]
    )
    integral_dense_grid_pdf_error /= (lims[0, 1] - lims[0, 0]) * (
        lims[1, 1] - lims[1, 0]
    )

    # DEBUGGING
    print("integral of mcmc kde error", integral_mcmc_kde_error)
    print("integral of dense grid pdf error", integral_dense_grid_pdf_error)

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

    # Assert that the errors are "small"
    # TODO: Giving a meaning to the hreshold / error should be easy, because we know the true pdf and the pdfs are normalized
    # TODO: Then evaluate whether the threshold is set reasonable
    # TODO: The threshold should be adapted depending on how hard the problem is
    # and how many samples / grid points we have
    threshold = 0.05
    assert integral_mcmc_kde_error < threshold
    assert integral_dense_grid_pdf_error < threshold


# Run the inference if main
if __name__ == "__main__":
    test_inference_mcmc_dense_exact()
