from typing import Optional

import jax.numpy as jnp
import numpy as np

from eulerpi.core.model import ArtificialModelInterface, JaxModel


class Heat(JaxModel):

    param_dim = 3
    data_dim = 4

    CENTRAL_PARAM = np.array([0.5, 0.5, 0.5])
    PARAM_LIMITS = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(central_param, param_limits, name=name, **kwargs)

    def get_param_bounds(self):
        return self.param_limits

    @classmethod
    def forward(self, param: np.ndarray) -> np.ndarray:
        solution = self.perform_simulation(self, param)
        # return the solution at four evaluation points
        eval_points = jnp.multiply(
            jnp.array(
                [
                    [0.25, 0.25, 0.25],
                    [0.25, 0.75, 0.25],
                    [0.25, 0.25, 0.75],
                    [0.25, 0.75, 0.75],
                ]
            ),
            jnp.array(
                [solution.shape[0], solution.shape[1], solution.shape[2]]
            ),
        ).astype(int)
        sim_res = jnp.array(
            solution[eval_points[:, 0], eval_points[:, 1], eval_points[:, 2]]
        )
        return sim_res

    def perform_simulation(self, param: np.ndarray) -> np.ndarray:
        # set up physical properties
        time_span = np.array([0, 0.1])
        plate_length = np.array([1, 1])

        # define the grid
        num_grid_points = 20
        dx = plate_length[0] / num_grid_points
        dy = plate_length[1] / num_grid_points
        safety_factor = 1.25
        dt = min(dx, dy) ** 2 / (
            safety_factor * 4 * jnp.max(self.PARAM_LIMITS[:, 1])
        )
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

        # solve numerically # TODO implement anisotropic
        for n in range(0, len(t) - 1):
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

    def jacobian(self, param: np.ndarray) -> np.ndarray:
        pass


class HeatArtificial(Heat, ArtificialModelInterface):

    CENTRAL_PARAM = np.array([0.5])
    PARAM_LIMITS = np.array([[0.0, 1.0]])

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(central_param, param_limits, name=name, **kwargs)

    def generate_artificial_params(self, num_samples: int) -> np.ndarray:
        lower_bound = self.param_limits[:, 0]
        upper_bound = self.param_limits[:, 1]
        true_param_sample = lower_bound + (
            upper_bound - lower_bound
        ) * np.random.rand(num_samples, self.param_dim)

        return true_param_sample
