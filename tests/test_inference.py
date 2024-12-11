import jax.scipy.stats as jstats
import matplotlib.pyplot as plt
import numpy as np

from eulerpi.core.dense_grid import generate_regular_grid
from eulerpi.core.inference import InferenceType, inference
from eulerpi.core.kde import calc_kernel_width, eval_kde_gauss
from eulerpi.core.result_managers import OutputWriter, ResultReader
from eulerpi.core.result_managers.result_manager_utils import get_run_path
from eulerpi.examples.simple_models import LinearODE


# WARNING: The following code only works for the simplest case. Equidistant grid, same number of points in each dimension, ...
def integrate(z, x, y):
    # Integrate the function over the grid
    integral = np.trapz(np.trapz(z, y, axis=0), x, axis=0)
    return integral


# TODO: Generalize, currently only works for dense vs mcmc
def test_inference_mcmc_dense_exact(
    num_data_points=1000,
    num_steps=1500,
    num_runs=2,  # Please keep this higher than 1. This tests is also responsible for testing whether the MCMC sampler and result_manager can concatenate multiple runs.
    num_grid_points=50,
):
    # define the model
    model = LinearODE()

    # generate artificial data
    params = model.generate_artificial_params(num_data_points)
    data = model.generate_artificial_data(params)

    # run EPI with all inference types
    result_params, result_pushforward_evals, result_densities = {}, {}, {}
    full_slice = np.arange(model.param_dim)
    for inference_type in InferenceType._member_map_.values():
        output_writer = OutputWriter(model.name, str(inference_type))
        if InferenceType(inference_type) == InferenceType.MCMC:
            _, _, _, result_reader = inference(
                model,
                data,
                inference_type,
                output_writer=output_writer,
                num_steps=num_steps,
                num_runs=num_runs,
            )
            # Take every second sample and skip the first 5% of the chain
            (
                result_params[inference_type],
                result_pushforward_evals[inference_type],
                result_densities[inference_type],
            ) = result_reader.load_inference_results(
                num_steps * num_runs // 20, 2
            )
        elif InferenceType(inference_type) == InferenceType.DENSE_GRID:
            _, _, _, result_reader = inference(
                model,
                data,
                inference_type,
                output_writer=output_writer,
                num_grid_points=num_grid_points,
            )
            (
                result_params[inference_type],
                result_pushforward_evals[inference_type],
                result_densities[inference_type],
            ) = result_reader.load_inference_results(full_slice)
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

    mcmc_params = result_params[InferenceType.MCMC]
    mcmc_kde = eval_kde_gauss(
        mcmc_params, grid, calc_kernel_width(mcmc_params)
    )

    dense_grid_pdf = result_densities[InferenceType.DENSE_GRID]

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

    def l2_error(x, y, error):
        return np.sqrt(integrate(to2d(error**2), x, y))

    # Calculate the L2 error
    integral_mcmc_kde_error = l2_error(x, y, mcmc_kde_error)
    integral_dense_grid_pdf_error = l2_error(x, y, dense_grid_pdf_error)

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
    threshold = 0.02
    assert integral_mcmc_kde_error < threshold
    assert integral_dense_grid_pdf_error < threshold


def test_thinning_and_burn_in():
    """Test if the thinning and burn in works as expected"""

    model = LinearODE()
    # generate artificial data
    params = model.generate_artificial_params(1000)
    data = model.generate_artificial_data(params)

    # run EPI with one trivial slice
    slice = np.array([0])
    num_steps = 1000
    num_walkers = 4
    num_runs = 2
    num_burn_in_samples = 100
    thinning_factor = 4

    # MCMC inference
    overall_params, pushforward_evals, density_evals, result_reader = (
        inference(
            model=model,
            data=data,
            slice=slice,
            inference_type=InferenceType.MCMC,
            num_steps=num_steps,
            num_walkers=num_walkers,
            num_burn_in_samples=num_burn_in_samples,
            thinning_factor=thinning_factor,
            num_runs=num_runs,
            run_name="test_thinning_and_burn_in",
        )
    )

    # Check if the thinning and burn in results in the expected shapes
    num_total_samples = (
        num_steps * num_walkers * num_runs - num_walkers * num_burn_in_samples
    ) // thinning_factor

    assert overall_params.shape[1] == slice.shape[0]
    assert pushforward_evals.shape[1] == model.data_dim
    assert density_evals.shape[1] == 1
    assert overall_params.shape[0] == num_total_samples
    assert pushforward_evals.shape[0] == num_total_samples
    assert density_evals.shape[0] == num_total_samples

    # create some artificial runs to test the thinning and burn in
    artificial_test_data_run = (
        np.arange(num_walkers * num_steps) // thinning_factor
    )
    np.savetxt(
        get_run_path(model.name, "test_thinning_and_burn_in")
        + "/Params/params_0.csv",
        artificial_test_data_run,
        delimiter=",",
    )
    artificial_test_data_run += 1000
    np.savetxt(
        get_run_path(model.name, "test_thinning_and_burn_in")
        + "/Params/params_1.csv",
        artificial_test_data_run,
        delimiter=",",
    )

    result_reader = ResultReader(model.name, "test_thinning_and_burn_in")

    (
        overall_params,
        pushforward_evals,
        density_evals,
    ) = result_reader.load_inference_results()
    # check if the correct samples where burned and thinned
    assert overall_params.shape[1] == slice.shape[0]
    assert pushforward_evals.shape[1] == model.data_dim
    assert density_evals.shape[1] == 1
    assert (
        overall_params.shape[0]
        == pushforward_evals.shape[0]
        == density_evals.shape[0]
        == num_total_samples
    )
    assert (
        overall_params.shape[0] - num_burn_in_samples
    ) % thinning_factor == 0


# Run the inference if main
if __name__ == "__main__":
    test_inference_mcmc_dense_exact()
