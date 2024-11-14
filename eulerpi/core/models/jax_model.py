from functools import partial
from typing import Optional, Tuple

import numpy as np
from jax import jacrev, jit, vmap

from .base_model import BaseModel
from .jax_extension import value_and_jacrev


def add_autodiff(_cls):
    """
    Decorator to automatically create the jacobian method based on the forward method.
    Additionally it jit compiles the forward and jacobian method with jax.

    Args:
        _cls: The class to decorate.

    Returns:
        The decorated class with the jacobian method and the forward and jacobian method jit compiled with jax.

    """
    _cls.init_fw_and_bw()
    return _cls


class JaxModel(BaseModel):
    """The JaxModel is a base class for models using the JAX library.

    It automatically creates the jacobian method based on the forward method.
    Additionally it jit compiles the forward and jacobian method with jax for faster execution.

    .. note::

        To use this class you have to implement your forward method using jax, e. g. jax.numpy.
        Dont overwrite the __init__ method of JaxModel without calling the super constructor.
        Else your forward method wont be jitted.
    """

    def __init__(
        self,
        central_param: np.ndarray,
        param_limits: np.ndarray,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            central_param=central_param,
            param_limits=param_limits,
            name=name,
            **kwargs,
        )
        # TODO: Check performance implications of not setting this at the class level but for each instance.
        type(self).forward = partial(JaxModel.forward_method, self)

    def __init_subclass__(cls, **kwargs):
        """Automatically create the jacobian method based on the forward method for the subclass."""
        return add_autodiff(super().__init_subclass__(**kwargs))

    @classmethod
    def init_fw_and_bw(cls):
        """Calculates the jitted methods for the subclass(es).
        It is an unintended sideeffect that this happens for all intermediate classes also.
        E.g. for: class CoronaArtificial(Corona)
        """
        cls.fw = jit(cls.forward)
        cls.bw = jit(jacrev(cls.forward))
        cls.vj = jit(value_and_jacrev(cls.forward))

    @staticmethod
    def forward_method(self, param: np.ndarray) -> np.ndarray:
        """This method is called by the jitted forward method. It is not intended to be called directly.

        Args:
            param(np.ndarray): The parameter for which the data should be generated.

        Returns:
            np.ndarray: The data generated from the parameter.

        """
        return type(self).fw(param)

    def jacobian(self, param: np.ndarray) -> np.ndarray:
        """Jacobian of the forward pass with respect to the parameters.

        Args:
            param(np.ndarray): The parameter for which the jacobian should be evaluated.

        Returns:
            np.ndarray: The jacobian for the variables returned by the :func:`~eulerpi.core.models.BaseModel.forward` method with respect to the parameters.

        """
        return type(self).bw(param)

    def forward_and_jacobian(
        self, param: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the jacobian and the forward pass of the model at the same time. This can be more efficient than calling the :func:`~eulerpi.core.models.BaseModel.forward` and :func:`~eulerpi.core.models.BaseModel.jacobian` methods separately.

        Args:
            param(np.ndarray): The parameter for which the jacobian should be evaluated.

        Returns:
            typing.Tuple[np.ndarray, np.ndarray]: The data and the jacobian for a given parameter.

        """
        return type(self).vj(param)

    def forward_vectorized(self, params: np.ndarray) -> np.ndarray:
        return vmap(self.forward, in_axes=0)(params)
