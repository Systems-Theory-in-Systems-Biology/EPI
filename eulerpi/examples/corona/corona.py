from typing import Optional

import diffrax as dx
import jax.numpy as jnp
import numpy as np

from eulerpi import logger
from eulerpi.core.model import ArtificialModelInterface, JaxModel


class Corona(JaxModel):
    """Describes the dynamics of the corona virus.

    .. math::
        :nowrap:

        \\begin{eqnarray}
            \\frac{d[S]}{dt} = & -q_1[S][I] \\\\
            \\frac{d[E]}{dt} = & q_1[S][I] - q_2[E] \\\\
            \\frac{d[I]}{dt}= & q_2[E] - q_3[I] \\\\
            \\frac{d[R]}{dt}= & q_3[I]
        \\end{eqnarray}

    subject to

    .. math::

        {\\left([S](t=0), \\ [E](t=0), \\ [I](t=0), \\ [R](t=0)\\right)}^\\intercal= \\left(999, \\ 0, \\ 1, \\ 0\\right)^\\intercal,

    .. note::

        * ODE Solver: To solve the ODE problem the jax based ode solver library :code:`diffrax` is used: https://github.com/patrick-kidger/diffrax.
        * Automatic Differentiation: The derivatives are calculated automatically with jax by deriving from the class :py:class:`~eulerpi.core.model.JaxModel`,
          which automatically calculates sets :py:meth:`~eulerpi.core.model.Model.jacobian`.
        * JIT compilation: Inheriting from :py:class:`~eulerpi.core.model.JaxModel` also enables jit compilation / optimization for the forward and jacobian method.
          This usually results in a significant execution speedup. It also allows to run your model on the gpu.
    """

    param_dim = 3
    data_dim = 4

    PARAM_LIMITS = np.array([[-4.5, 0.5], [-2.0, 3.0], [-2.0, 3.0]])
    CENTRAL_PARAM = np.array([-1.8, 0.0, 0.7])

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(central_param, param_limits, name=name, **kwargs)

    @classmethod
    def forward(cls, log_param):
        param = jnp.power(10, log_param)
        xInit = jnp.array([999.0, 0.0, 1.0, 0.0])

        def rhs(t, x, param):
            return jnp.array(
                [
                    -param[0] * x[0] * x[2],
                    param[0] * x[0] * x[2] - param[1] * x[1],
                    param[1] * x[1] - param[2] * x[2],
                    param[2] * x[2],
                ]
            )

        term = dx.ODETerm(rhs)
        solver = dx.Kvaerno5()
        saveat = dx.SaveAt(ts=[0.0, 1.0, 2.0, 5.0, 15.0])
        stepsize_controller = dx.PIDController(rtol=1e-5, atol=1e-5)

        try:
            ode_sol = dx.diffeqsolve(
                term,
                solver,
                t0=0.0,
                t1=15.0,
                dt0=0.1,
                y0=xInit,
                args=param,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
            )
            return ode_sol.ys[1:5, 2]

        except Exception as e:
            logger.warning("ODE solution not possible!", exc_info=e)
            return np.array([-np.inf, -np.inf, -np.inf, -np.inf])


class CoronaArtificial(Corona, ArtificialModelInterface):
    PARAM_LIMITS = np.array([[-2.5, -1.0], [-0.75, 0.75], [0.0, 1.5]])

    def __init__(
        self,
        central_param: np.ndarray = Corona.CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(central_param, param_limits, name=name, **kwargs)

    def generate_artificial_params(self, num_samples):
        lower_bound = np.array([-1.9, -0.1, 0.6])
        upper_bound = np.array([-1.7, 0.1, 0.8])

        true_param_sample = lower_bound + (
            upper_bound - lower_bound
        ) * np.random.rand(num_samples, 3)

        return true_param_sample
