import math
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax.lax import fori_loop

from eulerpi.core.model import ArtificialModelInterface, JaxModel


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
    with the thermal conductivity matrix :math:`\\kappa` and the temperature :math:`u`.
    Inference is performed on the entries of :math:`\\kappa`: param[0] = :math:`\\kappa_{11}`, param[1] = :math:`\\kappa_{22}`, param[2] = :math:`\\kappa_{12}`.
    Spatial discretization uses a finite difference scheme with a uniform grid, time stepping is done using the explicit Euler method.
    """

    param_dim = 3
    data_dim = 5
    evaluation_points = jnp.array(
        [
            [0.25, 0.25, 1],
            [0.75, 0.25, 1],
            [0.5, 0.5, 1],
            [0.25, 0.75, 1],
            [0.75, 0.75, 1],
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
    def forward(self, param: np.ndarray) -> np.ndarray:
        """Forward method for the heat model. Yields the solution of the anisotropic heat conduction equation at time :math:`\\t=0.1`
        in five spatial points, which are arranged similar to the number "five" on a dice.

        Args:
            param (np.ndarray): Entries of the conductivity matrix: param[0] = :math:`\\kappa_{11}`, param[1] = :math:`\\kappa_{22}`, param[2] = :math:`\\kappa_{12}`

        Returns:
            np.ndarray: The solution of the anisotropic heat conduction equation at time :math:`\\t=0.1`
        in five spacial points, which are arranged similar to the number "five" on a dice.
        """

        solution = self.perform_simulation(self, param)
        # return the solution at four evaluation points
        evaluation_indices = jnp.multiply(
            self.evaluation_points,
            jnp.array(
                [solution.shape[0], solution.shape[1], solution.shape[2]]
            ),
        ).astype(int)
        sim_res = jnp.array(
            solution[
                evaluation_indices[:, 0],
                evaluation_indices[:, 1],
                evaluation_indices[:, 2],
            ]
        )
        return sim_res

    def param_is_within_domain(self, param: np.ndarray) -> bool:
        """Checks whether a parameter is within the parameter domain of the model.
        This condition stems from thermodynamical considerations.

        Args:
            param(np.ndarray): The parameter to check.

        Returns:
            bool: True if the parameter is within the limits.

        """
        return param[0] * param[1] > param[2] ** 2

    def perform_simulation(self, param: np.ndarray) -> np.ndarray:
        """Performs a simulation of the heat equation with the given parameters.

        Args:
            param (np.ndarray): Entries of the conductivity matrix: param[0] = :math:`\\kappa_{11}`, param[1] = :math:`\\kappa_{22}`, param[2] = :math:`\\kappa_{12}`

        Returns:
            np.ndarray: An array containing the solution of the anisotropic heat conduction equation, where the first two indices correspond to the x and y coordinates, respectively, and the third index corresponds to the time.
        """
        # set up physical properties
        time_span = np.array([0, 0.1])
        plate_length = np.array([1, 1])

        # define the grid
        num_grid_points = 20
        dx = plate_length[0] / num_grid_points
        dy = plate_length[1] / num_grid_points

        # determine the time step size: this uses the stability condition for the explicit Euler method and parabolic problems, where
        # dt <= dx * dy / safety_factor * (4 * kappa_max), where kappa_max is the maximum eigenvalue of the thermal conductivity matrix and safety_factor >= 1.
        trace_kappa = self.PARAM_LIMITS[0, 1] + self.PARAM_LIMITS[1, 1]
        det_kappa = (
            self.PARAM_LIMITS[0, 1] * self.PARAM_LIMITS[1, 1]
            - self.PARAM_LIMITS[2, 1] ** 2
        )
        # Compute the maximum eigenvalue of the thermal conductivity matrix using trace and determinant:
        kappa_max = 0.5 * (
            trace_kappa + math.sqrt(trace_kappa**2 - 4 * det_kappa)
        )
        safety_factor = 1.25
        dt = min(dx, dy) ** 2 / (safety_factor * 4 * kappa_max)
        x = jnp.linspace(0, 1, num_grid_points)
        y = jnp.linspace(0, 1, num_grid_points)
        t = jnp.arange(time_span[0], time_span[1] + dt, dt)
        t = t.at[-1].set(time_span[1])

        # initial solution
        u = jnp.empty((len(x), len(y), len(t)))

        # define the initial condition
        u_init = jnp.zeros((len(x), len(y)))
        u = u.at[:, :, 0].set(u_init)

        # define the boundary conditions
        u_top = jnp.ones((len(x), len(t)))
        u_bottom = jnp.zeros((len(x), len(t)))
        u_left = jnp.ones((len(y), len(t)))
        u_right = jnp.zeros((len(y), len(t)))

        u = u.at[:, 0, :].set(u_bottom)
        u = u.at[:, -1, :].set(u_top)
        u = u.at[0, :, :].set(u_left)
        u = u.at[-1, :, :].set(u_right)

        # solve numerically
        # body function for the for loop:
        def integrate_time_step(n, u):
            du_dx2 = (
                u[2:, 1:-1, n] - 2 * u[1:-1, 1:-1, n] + u[:-2, 1:-1, n]
            ) / dx**2
            du_dy2 = (
                u[1:-1, 2:, n] - 2 * u[1:-1, 1:-1, n] + u[1:-1, :-2, n]
            ) / dy**2
            du_dx_dy = (
                u[2:, 2:, n] - u[2:, :-2, n] - u[:-2, 2:, n] + u[:-2, :-2, n]
            ) / (4 * dx * dy)
            u = u.at[1:-1, 1:-1, n + 1].set(
                (
                    param[0] * du_dx2
                    + param[1] * du_dy2
                    + 2 * param[2] * du_dx_dy
                )
                * dt
                + u[1:-1, 1:-1, n]
            )
            return u

        u = fori_loop(0, len(t) - 1, integrate_time_step, u)
        return u

    def jacobian(self, param: np.ndarray) -> np.ndarray:
        pass


class HeatArtificial(Heat, ArtificialModelInterface):

    CENTRAL_PARAM = np.array([1.5, 1.5, 0.5])
    PARAM_LIMITS = np.array([[1.0, 2.0], [1.0, 2.0], [0.0, 1.0]])

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
