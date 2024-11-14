import inspect
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseModel(ABC):
    """The base class for all models using the EPI algorithm.

    Args:
        central_param(np.ndarray): The central parameter for the model. (Default value = None)
        param_limits(np.ndarray): Box limits for the parameters. The limits are given as a 2D array with shape (param_dim, 2). The parameter limits are used as limits as well as for the movement policy for MCMC sampling, and as boundaries for the grid when using grid-based inference. Overwrite the function param_is_within_domain if the domain is more complex than a box - the grid will still be build based on param_limits, but actual model evaluations only take place within the limits specified in param_is_within_domain. (Default value = None)
        name(str): The name of the model. The class name is used if no name is given. (Default value = None)

    .. note::
        Examples of model implementations can be found in the :doc:`Example Models </examples>`.
    """

    param_dim: Optional[int] = (
        None  #: The dimension of the parameter space of the model. It must be defined in the subclass.
    )
    data_dim: Optional[int] = (
        None  #: The dimension of the data space of the model. It must be defined in the subclass.
    )

    def __init_subclass__(cls, **kwargs):
        """Check if the required attributes are set."""
        if not inspect.isabstract(cls):
            for required in (
                "param_dim",
                "data_dim",
            ):
                if not getattr(cls, required):
                    raise AttributeError(
                        f"Can't instantiate abstract class {cls.__name__} without {required} attribute defined"
                    )
        return cls

    def __init__(
        self,
        central_param: np.ndarray,
        param_limits: np.ndarray,
        name: Optional[str] = None,
    ) -> None:
        self.central_param = central_param
        self.param_limits = param_limits

        self.name = name or self.__class__.__name__

    @abstractmethod
    def forward(self, param: np.ndarray) -> np.ndarray:
        """Executed the forward pass of the model to obtain data from a parameter.

        Args:
            param(np.ndarray): The parameter for which the data should be generated.

        Returns:
            np.ndarray: The data generated from the parameter.

        Examples:

        .. code-block:: python

            import numpy as np
            from eulerpi.examples.heat import Heat
            from eulerpi.core.models import JaxModel
            from jax import vmap

            # instantiate the heat model
            model = Heat()

            # define a 3D example parameter for the heat model
            example_param = np.array([1.4, 1.6, 0.5])

            # the forward simulation is achieved by using the forward method of the model
            sim_result = model.forward(example_param)

            # in a more realistic scenario, we would like to perform the forward pass on multiple parameters at once
            multiple_params = np.array([[1.5, 1.5, 0.5],
                                        [1.4, 1.4, 0.6],
                                        [1.6, 1.6, 0.4],
                                        model.central_param,
                                        [1.5, 1.4, 0.4]])

            # try to use jax vmap to perform the forward pass on multiple parameters at once
            if isinstance(model, JaxModel):
                multiple_sim_results = vmap(model.forward, in_axes=0)(multiple_params)

            # if the model is not a jax model, we can use numpy vectorize to perform the forward pass
            else:
                multiple_sim_results = np.vectorize(model.forward, signature="(n)->(m)")(multiple_params)

        """
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, param: np.ndarray) -> np.ndarray:
        """Evaluates the jacobian of the :func:`~eulerpi.core.models.BaseModel.forward` method.

        Args:
            param(np.ndarray): The parameter for which the jacobian should be evaluated.

        Returns:
            np.ndarray: The jacobian for the variables returned by the :func:`~eulerpi.core.models.BaseModel.forward` method with respect to the parameters.

        Examples:

        .. code-block:: python

            import numpy as np
            from eulerpi.examples.heat import Heat
            from eulerpi.core.models import JaxModel
            from jax import vmap

            # instantiate the heat model
            model = Heat()

            # define a 3D example parameter for the heat model
            example_param = np.array([1.4, 1.6, 0.5])

            sim_jacobian = model.jacobian(example_param)

            # Similar to the forward pass, also the evaluation of the jacobian can be vectorized.
            # This yields a 3D array of shape (num_params, data_dim, param_dim) = (4,5,3) in this example.

            multiple_params = np.array([[1.5, 1.5, 0.5],
                                        [1.4, 1.4, 0.6],
                                        model.central_param,
                                        [1.5, 1.4, 0.4]])

            # try to use jax vmap for vectorization if possible
            if isinstance(model, JaxModel):
                multiple_sim_jacobians = vmap(model.jacobian, in_axes=0)(multiple_params)

            # if the model is not a jax model, we can use numpy vectorize to vectorize
            else:
                multiple_sim_jacobians = np.vectorize(model.jacobian, signature="(n)->(m)")(multiple_params)

        """
        raise NotImplementedError

    def forward_and_jacobian(
        self, param: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the jacobian and the forward pass of the model at the same time. If the method is not overwritten in a subclass it,
        it simply calls :func:`~eulerpi.core.models.BaseModel.forward` and :func:`~eulerpi.core.models.BaseModel.jacobian`.
        It can be vectorized in the same way as the forward and jacobian methods.

        Args:
            param(np.ndarray): The parameter for which the jacobian should be evaluated.

        Returns:
            typing.Tuple[np.ndarray, np.ndarray]: The data generated from the parameter and the jacobian for the variables returned by the :func:`~eulerpi.core.models.BaseModel.forward` method with respect to the parameters.

        """

        return self.forward(param), self.jacobian(param)

    def param_is_within_domain(self, param: np.ndarray) -> bool:
        """Checks whether a parameter is within the parameter domain of the model.
        Overwrite this function if your model has a more complex parameter domain than a box. The param_limits are checked automatically.

        Args:
            param(np.ndarray): The parameter to check.

        Returns:
            bool: True if the parameter is within the limits.

        """
        return True

    def forward_vectorized(self, params: np.ndarray) -> np.ndarray:
        """A vectorized version of the forward function

        Args:
            params (np.ndarray): an array of parameters, shape (n, self.param_dim)

        Returns:
            np.ndarray: The data vector generated from the vector of parameters, shape (n, self.data_dim)
        """
        return np.vectorize(self.forward, signature="(n)->(m)")(params)
