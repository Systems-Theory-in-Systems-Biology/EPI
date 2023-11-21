"""Basic plotting of eulerpi sampling results.

This module provides a basic plotting functionality to visualise sampling results for eulerpi.
Uses burn_in and thinning accordinng to the simulation settings.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axes

from eulerpi.core.kde import calc_kernel_width, eval_kde_gauss
from eulerpi.core.model import Model
from eulerpi.core.result_manager import ResultManager

# general plotting function for joint runs


def sample_violin_plot(
    model: Model,
    run_name: str = "default_run",
    what_to_plot: str = "param",
    artificial_bool: bool = False,
    credibility_level: float = 0.95,
    num_vertical_grid_points: int = 100,
    axis_labels: Optional[list[str]] = None,
) -> axes:

    """Creates an overview figure with one violin plot for each marginal distribution.
       Can be used for parameters and simulation results and compares true and inferred values when possible.

    Args:
        model(Model): The model describing the mapping from parameters to data.
        run_name(str): The name of the inference run. (Default value = "default_run")
        what_to_plot(str): Choose between "param" and "data" to respectively visualize either the model parameters or output. (Default value = "param")
        artificial_bool(bool): Indicates whether there is true (reference) parameter information available. (Default value = "False")
        credibility_level(float): Defines the probability mass (between 0 and 1) that is included within each of the violin plots. Choose 1 if you do not wand any cut-off. (Default value = 0.95)
        num_vertical_grid_points(int): Defines the resolution of the vertical violin plots. Can be increased for smoother plots or decresed for faster runtime. (default value = 100)
        axis_labels(list[str], optional): The labels depicted on the ordinate of the plot. Its size needs to be identical with the dimensionality of the plotted distribution. (Default value = None)

    Returns:
        axes: The overview figure with all violin plots as a matplotlib axes object.
    """

    # set figure font and color (also depending on what to plot)
    plt.rcParams.update({"font.size": 16})
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    if (what_to_plot == "param") and (not artificial_bool):
        true_bool = True
    else:
        true_bool = False

    if what_to_plot == "param":
        dim = model.param_dim
        variable_name = "Q"
        colorOrig = np.array([132.0, 143.0, 162.0]) / 255.0
        colorAppr = np.array([5.0, 142.0, 217.0]) / 255.0

    elif what_to_plot == "data":
        dim = model.data_dim
        variable_name = "Y"
        colorOrig = np.array([255.0, 147.0, 79.0]) / 255.0
        colorAppr = np.array([204.0, 45.0, 53.0]) / 255.0

    else:
        raise ValueError(
            "This function only supports plotting of model paramters (what_to_plot = 'param') or model outputs and date (what_to_plot = 'data')."
        )

    color3 = np.array([45.0, 49.0, 66.0]) / 255.0
    color4 = np.array([255.0, 218.0, 174.0]) / 255.0

    # define the locations and extends of the violin plots on the abscissa
    unit_locations = np.linspace(1, 2 * dim - 1, dim) / (2.0 * dim)
    envelope_width = 1.0 / (dim + 1)

    # load the results from the inference sampling
    rm = ResultManager(model_name=model.name, run_name=run_name)

    (
        reconstructed_param_sample,
        reconstructed_data_sample,
        _,
    ) = rm.load_slice_inference_results(
        slice=np.linspace(0, model.param_dim - 1, model.param_dim, dtype=int)
    )

    if what_to_plot == "param":
        reconstructed_sample = reconstructed_param_sample
    elif what_to_plot == "data":
        reconstructed_sample = reconstructed_data_sample

    # determine upper and lower plot bounds according to the credibility levels and what to plot
    if true_bool:
        true_sample = np.loadtxt(
            "Data/"
            + model.name
            + "/true_"
            + str(what_to_plot)
            + "_sample.csv",
            delimiter=",",
        )

        true_data_upper_percentile = np.percentile(
            true_sample, 100.0 * credibility_level, axis=0
        )
        true_data_lower_percentile = np.percentile(
            true_sample, 100.0 * (1 - credibility_level), axis=0
        )

        max_val = np.amax(true_data_upper_percentile)
        min_val = np.amin(true_data_lower_percentile)

    else:
        reconstructed_data_upper_percentile = np.percentile(
            reconstructed_sample, 100.0 * credibility_level, axis=0
        )
        reconstructed_data_lower_percentile = np.percentile(
            reconstructed_sample, 100.0 * (1 - credibility_level), axis=0
        )

        max_val = np.amax(reconstructed_data_upper_percentile)
        min_val = np.amin(reconstructed_data_lower_percentile)

    # create single figure with variable width
    fig, ax = plt.subplots(figsize=(2 * dim, 6))

    # set the plot axis limits and labels
    ax.set_ylim(min_val, max_val)
    x_labels = axis_labels or [
        r"$\mathcal{" + variable_name + "}_{" + str(d + 1) + "}$"
        for d in range(dim)
    ]
    ax.set_xticks(unit_locations, x_labels)

    # plot the vertical axes for all violin plots
    ax.vlines(unit_locations, min_val, max_val, color=color3, linewidth=1.0)

    # create one shared grid for all KDEs
    vertical_grid = np.transpose(
        np.array([np.linspace(min_val, max_val, num_vertical_grid_points)])
    )

    # also create a 1d array for the param grid
    vertical_grid_1d = np.squeeze(np.asarray(vertical_grid))

    # loop over all dimensions of either the param or the data
    for i in range(dim):

        reconstructed_matrix = np.transpose(
            np.array([reconstructed_sample[:, i]])
        )

        # in case there is a reference for the plotted qunatity avaialbe
        if true_bool:
            # cast to 2d array
            true_matrix = np.transpose(np.array([true_sample[:, i]]))

            # calculate kernel width for KDE
            scales = calc_kernel_width(true_matrix)

            # evaluate KDEs over the grid
            true_KDE = eval_kde_gauss(true_matrix, vertical_grid, scales)

            # normalize the KDEs and caluculate their cumulative distribution
            true_KDE_norm_cumsum = np.cumsum(true_KDE / np.sum(true_KDE))

            # create boolean arrays to filter the KDEs for the specified credibility level
            true_konfidence_index = (
                true_KDE_norm_cumsum > (1 - credibility_level) / 2.0
            ) & (
                true_KDE_norm_cumsum
                < credibility_level + (1 - credibility_level) / 2.0
            )

            # calculate the maximum density of the KDEs and the corresponding incidence
            max_density = np.amax(true_KDE)
            max_density_argument = vertical_grid_1d[np.argmax(true_KDE)]

            # calculate violin envelopes for reference and reconstruction
            true_left_bound = (
                -0.5 * envelope_width / max_density * true_KDE
                + unit_locations[i]
            )
            true_right_bound = (
                0.5 * envelope_width / max_density * true_KDE
                + unit_locations[i]
            )

            # filter the violin envelopes for the specified credibility level
            true_left_bound_konf = true_left_bound[true_konfidence_index]
            true_right_bound_konf = true_right_bound[true_konfidence_index]

            # plot the filtered violin envelopes for the reference
            ax.plot(
                true_left_bound_konf,
                vertical_grid_1d[true_konfidence_index],
                linewidth=3.0,
                color=colorOrig,
            )
            ax.plot(
                true_right_bound_konf,
                vertical_grid_1d[true_konfidence_index],
                linewidth=3.0,
                color=colorOrig,
            )

            # close the envelopes by connecting the last and the first point
            for j in [0, -1]:
                ax.plot(
                    [true_left_bound_konf[j], true_right_bound_konf[j]],
                    [
                        vertical_grid_1d[true_konfidence_index][j],
                        vertical_grid_1d[true_konfidence_index][j],
                    ],
                    linewidth=3.0,
                    color=colorOrig,
                )

            # fill the violin envelopes

            ax.fill_betweenx(
                vertical_grid_1d[true_konfidence_index],
                true_left_bound_konf,
                true_right_bound_konf,
                color=colorOrig,
                label=r"$\Phi_\mathcal{" + variable_name + "}$"
                if i == 0
                else "",
                alpha=0.3,
            )

            # draw arrows to show the width of the violin envelopes
            ax.arrow(
                np.amin(true_left_bound),
                max_density_argument,
                np.amax(true_right_bound) - np.amin(true_left_bound),
                0,
                length_includes_head=True,
                color=color3,
                head_width=(max_val - min_val) / 100.0,
                head_length=0.02,
                linewidth=1.0,
            )

            ax.arrow(
                np.amax(true_right_bound),
                max_density_argument,
                -np.amax(true_right_bound) + np.amin(true_left_bound),
                0,
                length_includes_head=True,
                color=color3,
                head_width=(max_val - min_val) / 100.0,
                head_length=0.02,
                linewidth=1.0,
            )

            ax.text(
                unit_locations[i] + 0.01,
                max_density_argument + (max_val - min_val) / 50.0,
                "%.2f" % (np.round(max_density, 2)),
            )

        # in case of no reference, caluclate the kernel bandwidth from the reconstruction
        else:
            scales = calc_kernel_width(reconstructed_matrix)

        # repeat all plotting for the reconstruction
        reconstructed_KDE = eval_kde_gauss(
            reconstructed_matrix, vertical_grid, scales
        )

        if not true_bool:
            max_density = np.amax(reconstructed_KDE)
            max_density_argument = vertical_grid_1d[
                np.argmax(reconstructed_KDE)
            ]

        reconstructed_KDE_norm_cumsum = np.cumsum(
            reconstructed_KDE / np.sum(reconstructed_KDE)
        )
        reconstructed_konfidence_index = (
            reconstructed_KDE_norm_cumsum > (1 - credibility_level) / 2.0
        ) & (
            reconstructed_KDE_norm_cumsum
            < credibility_level + (1 - credibility_level) / 2.0
        )
        reconstructed_left_bound = (
            -0.5 * envelope_width / max_density * reconstructed_KDE
            + unit_locations[i]
        )
        reconstructed_right_bound = (
            0.5 * envelope_width / max_density * reconstructed_KDE
            + unit_locations[i]
        )
        reconstructed_left_bound_konf = reconstructed_left_bound[
            reconstructed_konfidence_index
        ]
        reconstructed_right_bound_konf = reconstructed_right_bound[
            reconstructed_konfidence_index
        ]

        ax.plot(
            reconstructed_left_bound_konf,
            vertical_grid_1d[reconstructed_konfidence_index],
            linewidth=3.0,
            color=colorAppr,
        )

        ax.plot(
            reconstructed_right_bound_konf,
            vertical_grid_1d[reconstructed_konfidence_index],
            linewidth=3.0,
            color=colorAppr,
        )

        for j in [0, -1]:
            ax.plot(
                [
                    reconstructed_left_bound_konf[j],
                    reconstructed_right_bound_konf[j],
                ],
                [
                    vertical_grid_1d[reconstructed_konfidence_index][j],
                    vertical_grid_1d[reconstructed_konfidence_index][j],
                ],
                linewidth=3.0,
                color=colorAppr,
            )

        ax.fill_betweenx(
            vertical_grid_1d[reconstructed_konfidence_index],
            reconstructed_left_bound_konf,
            reconstructed_right_bound_konf,
            color=colorAppr,
            label=r"$\Phi_{\hat{\mathcal{" + variable_name + "}}}$"
            if i == 0
            else "",
            alpha=0.3,
        )

        if not true_bool:
            # draw arrows to show the width of the violin envelopes
            ax.arrow(
                np.amin(reconstructed_left_bound),
                max_density_argument,
                np.amax(reconstructed_right_bound)
                - np.amin(reconstructed_left_bound),
                0,
                length_includes_head=True,
                color=color3,
                head_width=(max_val - min_val) / 50.0,
                head_length=0.02,
                linewidth=1.0,
            )
            ax.arrow(
                np.amax(reconstructed_right_bound),
                max_density_argument,
                -np.amax(reconstructed_right_bound)
                + np.amin(reconstructed_left_bound),
                0,
                length_includes_head=True,
                color=color3,
                head_width=(max_val - min_val) / 50.0,
                head_length=0.02,
                linewidth=1.0,
            )
            ax.text(
                unit_locations[i] + 0.01,
                max_density_argument + (max_val - min_val) / 50.0,
                "%.2f" % (np.round(max_density, 2)),
            )

    ax.legend()
    ax.set_xlim(0.0, 1.0)
    plt.tight_layout()

    return ax
