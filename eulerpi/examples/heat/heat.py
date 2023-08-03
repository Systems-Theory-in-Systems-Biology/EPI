from typing import Optional

import diffrax
import jax.numpy as jnp
import numpy as np

from eulerpi.core.model import ArtificialModelInterface, JaxModel


def heat_rhs(t: float, u: jnp.ndarray, args: tuple | list) -> jnp.ndarray:
    """Right hand side of the heat equation.

    Args:
        t (float): time at which the right hand side is evaluated
        u (jnp.ndarray): current solution of the heat equation
        args (tuple | list): tuple of the form (dx, dy, param) with dx, dy the spatial discretization and param the thermal conductivity matrix, given as a vector of length 3 with param[0] = kappa_11, param[1] = kappa_22, param[2] = kappa_12

    Returns:
        jnp.ndarray: right hand side of the heat equation
    """
    dx = args[0]
    dy = args[1]
    param = args[2]

    # use the central difference scheme to approximate the derivatives. Gradient preserves the size of the array by using one-sided differences. We throw away the boundary points later.
    du_dx = jnp.gradient(u, dx, axis=0)
    du_dx2 = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
    du_dy2 = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    du_dx_dy = jnp.gradient(du_dx, dy, axis=1)

    # compute the right hand side of the heat equation
    rhs = jnp.zeros(u.shape)
    rhs = rhs.at[1:-1, 1:-1].set(
        param[0] * du_dx2
        + param[1] * du_dy2
        + 2 * param[2] * du_dx_dy[1:-1, 1:-1]
    )
    return rhs


class Heat(JaxModel):
    """A two-dimensional anisotropic heat conduction equation model on a square domain with four dirichlet boundaries.
    The model is defined by the following partial differential equation on the square spacial domain :math:`\\Omega = [0, 1]^2` on the time interval :math:`[0, 0.1]`:
    .. math::
        \\frac{\\partial u}{\\partial t} = \\div \\left( \\kappa \\nabla u \\right)
    subject to
    .. math::
        u(x, y, t=0) = 0
    and
    .. math::
        u(0, y, t) = 1, \\quad u(1, y, t) = 0, \\quad u(x, 0, t) = 1, \\quad u(x, 1, t) = 0
    with the symmetric positive definite thermal conductivity matrix :math:`\\kappa` and the temperature :math:`u`.
    Inference is performed on the entries of :math:`\\kappa`: param[0] = :math:`\\kappa_{11}`, param[1] = :math:`\\kappa_{22}`, param[2] = :math:`\\kappa_{12}`.
    """

    t_end = 0.1
    plate_length = jnp.array([1.0, 1.0])
    num_grid_points = 20

    param_dim = 3
    data_dim = 5  # The values of the heat equation solution at five points are observed. See evaluation_points.

    evaluation_points = jnp.array(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.5, 0.5],
            [0.25, 0.75],
            [0.75, 0.75],
        ]
    )

    CENTRAL_PARAM = np.array([1.5, 1.5, 0.5])
    PARAM_LIMITS = np.array([[1.0, 2.0], [1.0, 2.0], [0.0, 1.0]])

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Default constructor for the heat model.

        Args:
            central_param (np.ndarray, optional): Central parameter within the parameter domain, where the density is non-zero. Defaults to CENTRAL_PARAM.
            param_limits (np.ndarray, optional): Maximum parameter limits for sampling or grid-based inference. The parameter domain is contained within the box given by param_limits. Defaults to PARAM_LIMITS.
            name (Optional[str], optional): Name for the model. Defaults to None means the model name is "heat".
        """
        super().__init__(central_param, param_limits, name=name, **kwargs)

    @classmethod
    def forward(cls, param: np.ndarray) -> np.ndarray:
        """Forward method for the heat model. Yields the solution of the anisotropic heat conduction equation at time :math:`\\t=0.1`
        in five spatial points, which are arranged similar to the number "five" on a dice.

        Args:
            param (np.ndarray): Entries of the conductivity matrix: param[0] = :math:`\\kappa_{11}`, param[1] = :math:`\\kappa_{22}`, param[2] = :math:`\\kappa_{12}`

        Returns:
            np.ndarray: The solution of the anisotropic heat conduction equation at time :math:`\\t=0.1`
        in five spacial points, which are arranged similar to the number "five" on a dice.
        """

        solution = cls.perform_simulation(kappa=param)

        # linearly interpolate the solution at the evaluation points, use own interpolation function as jax doesn't support scipy.interpolate.interp2d and doesn't provide a 2d interpolation function
        x = jnp.linspace(0, cls.plate_length[0], cls.num_grid_points)
        y = jnp.linspace(0, cls.plate_length[1], cls.num_grid_points)

        # compute the indices between which the evaluation points lie
        x_indices = jnp.searchsorted(x, cls.evaluation_points[:, 0]) - 1
        y_indices = jnp.searchsorted(y, cls.evaluation_points[:, 1]) - 1
        dx = cls.plate_length[0] / cls.num_grid_points
        dy = cls.plate_length[1] / cls.num_grid_points

        # interpolate the solution at the evaluation points
        solution_at_evaluation_points = (1 / (dx * dy)) * (
            solution[x_indices, y_indices]
            * (x[x_indices + 1] - cls.evaluation_points[:, 0])
            * (y[y_indices + 1] - cls.evaluation_points[:, 1])
            + solution[x_indices + 1, y_indices]
            * (cls.evaluation_points[:, 0] - x[x_indices])
            * (y[y_indices + 1] - cls.evaluation_points[:, 1])
            + solution[x_indices, y_indices + 1]
            * (x[x_indices + 1] - cls.evaluation_points[:, 0])
            * (cls.evaluation_points[:, 1] - y[y_indices])
            + solution[x_indices + 1, y_indices + 1]
            * (cls.evaluation_points[:, 0] - x[x_indices])
            * (cls.evaluation_points[:, 1] - y[y_indices])
        )
        return solution_at_evaluation_points

    def param_is_within_domain(self, param: np.ndarray) -> bool:
        """Checks whether a parameter is within the parameter domain of the model.
        This condition stems from thermodynamical considerations.

        Args:
            param(np.ndarray): The parameter to check.

        Returns:
            bool: True if the parameter is within the limits.

        """
        return param[0] * param[1] > param[2] ** 2

    @classmethod
    def perform_simulation(cls, kappa: np.ndarray) -> np.ndarray:
        """Performs a simulation of the heat equation with the given parameters.

        Args:
            kappa (np.ndarray): Entries of the conductivity matrix: kappa[0] = :math:`\\kappa_{11}`, kappa[1] = :math:`\\kappa_{22}`, kapp[2] = :math:`\\kappa_{12}`

        Returns:
            np.ndarray: An array containing the solution of the anisotropic heat conduction equation, where the first two indices correspond to the x and y coordinates, respectively. The time is fixed to the class variable t_end.
        """
        # import os
        # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

        # The grid
        x = jnp.linspace(0, cls.plate_length[0], cls.num_grid_points)
        y = jnp.linspace(0, cls.plate_length[1], cls.num_grid_points)
        dx = cls.plate_length[0] / cls.num_grid_points
        dy = cls.plate_length[1] / cls.num_grid_points

        def stable_time_step(dx, dy, kappa):
            trace_kappa = kappa[0] + kappa[1]
            det_kappa = kappa[0] * kappa[1] - kappa[2] ** 2
            # Compute the maximum eigenvalue of the thermal conductivity matrix using trace and determinant:
            kappa_max = 0.5 * (
                trace_kappa + jnp.sqrt(trace_kappa**2 - 4 * det_kappa)
            )
            safety_factor = 1.25
            dt = jnp.minimum(dx, dy) ** 2 / (safety_factor * 4 * kappa_max)
            return dt

        dt0 = stable_time_step(dx, dy, kappa)
        u0 = jnp.zeros((len(x), len(y)))
        # Set boundary conditions
        u0 = u0.at[0, :].set(jnp.ones(len(y)))  # left
        u0 = u0.at[-1, :].set(jnp.zeros(len(y)))  # right
        u0 = u0.at[:, 0].set(jnp.zeros(len(x)))  # bottom
        u0 = u0.at[:, -1].set(jnp.ones(len(x)))  # top

        # perform the time integration using diffrax
        term = diffrax.ODETerm(heat_rhs)
        solver = diffrax.Heun()
        saveat = diffrax.SaveAt(t0=False, t1=True)
        stepsize_controller = diffrax.PIDController(
            rtol=1e-5, atol=1e-5, dtmax=dt0
        )
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=cls.t_end,
            dt0=dt0,
            saveat=saveat,
            y0=u0,
            stepsize_controller=stepsize_controller,
            args=(dx, dy, kappa),
        )
        u = sol.ys[-1]  # The solution at the final time
        return u


class HeatArtificial(Heat, ArtificialModelInterface):

    CENTRAL_PARAM = np.array([1.5, 1.5, 0.5])
    PARAM_LIMITS = np.array([[1.0, 2.0], [1.0, 2.0], [0.0, 1.0]])
    num_grid_points = 20

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Default constructor for the artificial heat model.

        Args:
            central_param (np.ndarray, optional): Central parameter within the parameter domain, where the density is non-zero. Defaults to CENTRAL_PARAM.
            param_limits (np.ndarray, optional): Maximum parameter limits for sampling or grid-based inference. The parameter domain is contained within the box given by param_limits. Defaults to PARAM_LIMITS.
            name (Optional[str], optional): Name for the model. Defaults to None means the model name is "heat".
        """
        super().__init__(central_param, param_limits, name=name, **kwargs)

    def generate_artificial_params(
        self, num_samples: int, independent_params: bool = True
    ) -> np.ndarray:
        """Generates a set of viable parameter samples for the heat model.

        Args:
            num_samples (int): Number of samples to generate.
            independent_params (bool, optional): Whether the parameters should be independent. Defaults to True.

        Returns:
            np.ndarray: A set of viable parameter samples.
        """
        param_spans = self.param_limits[:, 1] - self.param_limits[:, 0]
        lower_bounds = self.param_limits[:, 0] + 0.2 * param_spans
        upper_bounds = self.param_limits[:, 1] - 0.2 * param_spans
        true_param_sample = lower_bounds + (
            upper_bounds - lower_bounds
        ) * np.random.beta(a=2, b=5, size=(num_samples, self.param_dim))
        return true_param_sample
