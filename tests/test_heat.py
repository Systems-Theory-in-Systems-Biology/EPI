import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from eulerpi.examples.heat import Heat, HeatArtificial


def test_heat_model():
    """Test the heat model and plot the solution of a simple model problem."""
    # define the model
    heat_model = Heat()

    # test the model
    u = heat_model.perform_simulation(np.array([0.5, 0.5, 0.25]))[:, :, -1]

    # build the grid
    y_1 = np.linspace(0, 1, 100)
    y_2 = np.linspace(0, 1, 100)
    Y_1, Y_2 = np.meshgrid(y_1, y_2)
    extent = [y_1[0], y_1[-1], y_2[0], y_2[-1]]

    evaluation_points = np.vstack([Y_1.ravel(), Y_2.ravel()]).T

    # Define the color map
    colors = [
        "#762a83",
        "#9970ab",
        "#c2a5cf",
        "#e7d4e8",
        "#d9f0d3",
        "#a6dba0",
        "#5aae61",
        "#1b7837",
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)

    # plot the KDE: draw the function
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    im = ax.imshow(u.T, origin="lower", cmap=cmap, extent=extent, aspect=1)

    # draw the contour lines
    cset = ax.contour(
        u.T,
        np.arange(np.min(u), np.max(u), (np.max(u) - np.min(u)) / 8),
        linewidths=2,
        color="k",
        extent=extent,
        aspect=1,
    )
    ax.clabel(cset, inline=True, fmt="%1.2f", fontsize=10)

    # draw the colorbar
    fig.colorbar(im, ax=ax, location="right")

    # add pretty stuff
    fig.suptitle(r"Solution of the heat equation")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    plt.show()


def test_heat_artificial():
    """Test the artificial data generation of the heat model."""
    # define the model
    heat_model = HeatArtificial()
    # generate some artificial params
    artificial_params = heat_model.generate_artificial_params(20)
    # check correct shape
    assert artificial_params.shape == (20, 3)
    # check that all params are within the domain
    assert np.all(heat_model.param_is_within_domain(artificial_params.T))
    # generate some artificial data
    artificial_data = heat_model.generate_artificial_data(artificial_params)
    # check correct shape
    assert artificial_data.shape == (20, 5)
