import importlib
from typing import Optional

import jax.numpy as jnp
import numpy as np

from eulerpi.core.model import ArtificialModelInterface, Model

# from functools import partial
# from jax import jit


class Temperature(Model):
    """The model describes the temperature :math:`y` in degree celsius at a given latitude :math:`q` in degree.


    .. math::

        s: [0^{\\circ}, 90^{\\circ}] \\rightarrow [-30, 30]

        q \\mapsto y = s(q) := 60 \\cdot \\cos{(q)} - 30

    :math:`q=0^{\\circ}` corresponds to the equator and :math:`q=90^{\\circ}` to the north pole.
    """

    param_dim = 1  #: The latitude of the location
    data_dim = 1  #: The temperature in degree celsius

    PARAM_LIMITS = np.array(
        [[0, np.pi / 2]]
    )  #: The latitude is between :math:`0^{\circ}` and :math:`90^{\circ}`
    CENTRAL_PARAM = np.array(
        [np.pi / 4.0]
    )  #: The central latitude is :math:`\pi/4 = 45^{\circ}`

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(central_param, param_limits, name=name, **kwargs)

    def forward(self, param):
        """Evaluates the temperature at the given latitude using the model equation

        .. math::

            y = 60 \\cdot \\cos{(q)} - 30
        """

        low_T = -30.0
        high_T = 30.0
        res = jnp.array(
            [low_T + (high_T - low_T) * jnp.cos(jnp.abs(param[0]))]
        )
        return res

    def jacobian(self, param):
        """Calculates the analytical jacobian of the model equation at the given latitude

        .. math::

            \\frac{dy}{dq} = -60 \\cdot \\sin{(q)}
        """
        return jnp.array([60.0 * jnp.sin(jnp.abs(param[0]))])


class TemperatureArtificial(Temperature, ArtificialModelInterface):
    def generate_artificial_params(self, num_data_points: int = -1):
        paramPath = importlib.resources.path(
            "eulerpi.examples.temperature", "TemperatureArtificialParams.csv"
        )
        true_param_sample = np.loadtxt(paramPath, delimiter=",", ndmin=2)
        return true_param_sample


class TemperatureWithFixedParams(Temperature):
    """The model describes the temperature :math:`y` in degree celsius at a given latitude :math:`q` in degree.

    .. note::

        * Additional fixed parameters: The model includes fixed parameters :code:`self.low_T=30.0` and :code:`self.high_T=30.0`.
          These fixed parameters are passed to the calc_forward function separately. You can create models with different parameters by
          creating several model objects.
          The best way to separate the outputs for the parametrized models is to pass a string based on the fixed_params to the attribute :py:attr:`run_name` of the :py:func:`~eulerpi.core.inference` function.
        * The functions :py:meth:`~eulerpi.examples.temperature.temperature.TemperatureWithFixedParams.calc_forward` is not strictly necessary.
          However it can help to make it work with jax.
    """

    def __init__(
        self,
        low_T: np.double = -30.0,
        high_T: np.double = 30.0,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.low_T = low_T
        self.high_T = high_T

    def forward(self, param):
        return self.calc_forward(param, self.high_T, self.low_T)

    def calc_forward(self, param, high_T, low_T):
        res = jnp.array(
            [low_T + (high_T - low_T) * jnp.cos(jnp.abs(param[0]))]
        )
        return res

    def jacobian(self, param):
        return self.calc_jacobian(param, self.high_T, self.low_T)

    def calc_jacobian(self, param, high_T, low_T):
        return jnp.array([(high_T - low_T) * jnp.sin(jnp.abs(param[0]))])
