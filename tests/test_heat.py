import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from eulerpi.examples.heat import Heat, HeatArtificial


def test_heat_model():
    """Test the heat model and plot the solution of a simple model problem."""
    # define the model
    heat_model = Heat()

    # test the model
    u = heat_model.perform_simulation(kappa=jnp.array([0.5, 0.5, 0.25]))

    # build the grid
    y_1 = np.linspace(0, 1, 100)
    y_2 = np.linspace(0, 1, 100)
    Y_1, Y_2 = np.meshgrid(y_1, y_2)
    extent = [y_1[0], y_1[-1], y_2[0], y_2[-1]]

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
        np.arange(0, 1, 0.1),
        # np.arange(np.min(u), np.max(u), (np.max(u) - np.min(u)) / 8),
        linewidths=2,
        extent=extent,
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
    n_params = 20
    # generate some artificial params
    artificial_params = heat_model.generate_artificial_params(n_params)
    # check correct shape
    assert artificial_params.shape == (n_params, heat_model.param_dim)
    # check that all params are within the domain
    assert np.all(heat_model.param_is_within_domain(artificial_params.T))
    # generate some artificial data
    artificial_data = heat_model.generate_artificial_data(artificial_params)
    # check correct shape
    assert artificial_data.shape == (n_params, heat_model.data_dim)


# TODO: possbly we should check this for all models in test_examples.py?
# This test specifically exists because we had problems with the backpropagation in the heat model, where we received only zeros.
def test_heat_jacobian():
    model = HeatArtificial()
    h = 1e-3
    res = model.forward(model.central_param)
    res1 = model.forward(model.central_param + np.array([h, 0, 0]))
    res2 = model.forward(model.central_param + np.array([0, h, 0]))
    res3 = model.forward(model.central_param + np.array([0, 0, h]))
    jac = model.jacobian(model.central_param)

    # compare with finite differences
    jac_fd = np.zeros((model.data_dim, model.param_dim))
    jac_fd[:, 0] = (res1 - res) / h
    jac_fd[:, 1] = (res2 - res) / h
    jac_fd[:, 2] = (res3 - res) / h
    assert np.allclose(jac, jac_fd, atol=1e-3)

    # Assert that the jacobian is not zero and that norm is larger than the same tolerance
    assert not np.allclose(jac, 0, atol=1e-5)
    assert np.linalg.norm(jac) > 1e-5
