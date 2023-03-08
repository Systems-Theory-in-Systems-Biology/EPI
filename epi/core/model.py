import inspect
import os
import typing
from abc import ABC, abstractmethod
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit, vmap

from epi.jax_extension import value_and_jacrev


class Model(ABC):
    """The base class for all models using the EPI algorithm.

    Attributes:
        param_dim(int): The dimension of the parameter space.
        data_dim(int): The dimension of the data space.

        defaultParamSamplingLimits(np.ndarray): The default limits for the parameters. The limits are given as a 2D array with shape (param_dim, 2).
        defaultcentral_param(np.ndarray): The default central parameter for the model.

        central_param(np.ndarray): The central parameter for the model.
        param_limits(np.ndarray): The limits for the parameters. The limits are given as a 2D array with shape (param_dim, 2).
        name(str): The name of the model.

    """

    param_dim = None
    data_dim = None

    defaultParamSamplingLimits = None
    defaultcentral_param = None

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
        central_param: np.ndarray = None,
        param_limits: np.ndarray = None,
        name: str = None,
    ) -> None:
        """Initializes the model with the given parameters.

        Args:
            central_param(np.ndarray): The central parameter for the model. (Default value = None)
            param_limits(np.ndarray): The limits for the parameters. The limits are given as a 2D array with shape (param_dim, 2). (Default value = None)
            name(str): The name of the model. The class name is used if no name is given. (Default value = None)
        """

        assert (
            central_param is not None or self.defaultcentral_param is not None
        )
        assert (
            param_limits is not None
            or self.defaultParamSamplingLimits is not None
        )

        self.central_param = (
            central_param if central_param else self.defaultcentral_param
        )
        self.param_limits = (
            param_limits if param_limits else self.defaultParamSamplingLimits
        )

        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

    @abstractmethod
    def forward(self, param: np.ndarray) -> np.ndarray:
        """Executed the forward pass of the model to obtain data from a parameter. You can also do equivalently :code:`model(param)`.

        Args:
            param(np.ndarray): The parameter for which the data should be generated.

        Returns:
            np.ndarray: The data generated from the parameter.

        """
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, param: np.ndarray) -> np.ndarray:
        """Evaluates the jacobian of the :func:`~epic.core.model.Model.forward` method.

        Args:
            param(np.ndarray): The parameter for which the jacobian should be evaluated.

        Returns:
            np.ndarray: The jacobian for the variables returned by the :func:`~epic.core.model.Model.forward` method with respect to the parameters.

        """
        raise NotImplementedError

    def valjac(
        self, param: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Evaluates the jacobian and the forward pass of the model at the same time. If the method is not overwritten in a subclass it,
        it simply calls :func:`~epic.core.model.Model.forward` and :func:`~epic.core.model.Model.jacobian`.

        Args:
            param(np.ndarray): The parameter for which the jacobian should be evaluated.

        Returns:
            typing.Tuple[np.ndarray, np.ndarray]: The data generated from the parameter and the jacobian for the variables returned by the :func:`~epic.core.model.Model.forward` method with respect to the parameters.

        """

        return self.forward(param), self.jacobian(param)

    def is_artificial(self) -> bool:
        """Determines whether the model provides artificial parameter and data sets.

        Returns:
            bool: True if the model inherits from the ArtificialModelInterface

        """
        return issubclass(self.__class__, ArtificialModelInterface)


class ArtificialModelInterface(ABC):
    """By inheriting from this interface you indicate that you are providing an artificial parameter dataset,
    and the corresponding artificial data dataset, which can be used to compare the results from epi with the ground truth.
    The comparison can be done using the plotEmceeResults.

    """

    @abstractmethod
    def generate_artificial_params(self, num_samples: int) -> np.ndarray:
        """This method must be overwritten an return an numpy array of num_samples parameters.

        Args:
            num_samples(int): The number of parameters to generate.

        Returns:
            np.ndarray: The generated parameters.

        Raises:
            NotImplementedError: If the method is not overwritten in a subclass.

        """
        raise NotImplementedError

    def generate_artificial_data(
        self,
        params: typing.Union[os.PathLike, str, np.ndarray],
    ) -> np.ndarray:
        """This method is called when the user wants to generate artificial data from the model.

        Args:
            params: typing.Union[os.PathLike, str, np.ndarray]: The parameters for which the data should be generated. Can be either a path to a file, a numpy array or a string.

        Returns:
            np.ndarray: The data generated from the parameters.

        Raises:
            TypeError: If the params argument is not a path to a file, a numpy array or a string.

        """
        if isinstance(params, str) or isinstance(params, os.PathLike):
            params = np.loadtxt(params, delimiter=",", ndmin=2)
        elif isinstance(params, np.ndarray) or isinstance(params, jnp.ndarray):
            pass
        else:
            raise TypeError(
                f"The params argument has to be either a path to a file or a numpy array. The passed argument was of type {type(params)}"
            )

        # try to use jax vmap to perform the forward pass on multiple parameters at once
        if isinstance(self, JaxModel):
            return vmap(self.forward, in_axes=0)(params)
        else:
            return np.vectorize(self.forward, signature="(n)->(m)")(params)


def autodiff(_cls):
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


class JaxModel(Model):
    """The JaxModel class automatically creates the jacobian method based on the forward method.
    Additionally it jit compiles the forward and jacobian method with jax.
    To use this class you have to implement your forward method using jax, e. g. jax.numpy.
    Dont overwrite the __init__ method of JaxModel without calling the super constructor.
    Else your forward method wont be jitted.

    """

    def __init__(self, name: str = None) -> None:
        """Constructor of the JaxModel class.

        Args:
            name: str: The name of the model. If None the name of the class is used.
        """
        super().__init__(name=name)
        # TODO: Check performance implications of not setting this at the class level but for each instance.
        type(self).forward = partial(JaxModel.forward_method, self)

    def __init_subclass__(cls, **kwargs):
        """Automatically create the jacobian method based on the forward method for the subclass."""
        return autodiff(super().__init_subclass__(**kwargs))

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
            np.ndarray: The jacobian for the variables returned by the :func:`~epic.core.model.Model.forward` method with respect to the parameters.

        """
        return type(self).bw(param)

    def valjac(
        self, param: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Evaluates the jacobian and the forward pass of the model at the same time. This can be more efficient than calling the :func:`~epic.core.model.Model.forward` and :func:`~epic.core.model.Model.jacobian` methods separately.

        Args:
            param(np.ndarray): The parameter for which the jacobian should be evaluated.

        Returns:
            typing.Tuple[np.ndarray, np.ndarray]: The data and the jacobian for a given parameter.

        """
        return type(self).vj(param)
