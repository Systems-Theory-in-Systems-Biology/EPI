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

    It contains three abstract methods which need to be implemented by subclasses

    Args:

    Returns:

    """

    paramDim = None
    dataDim = None

    defaultParamSamplingLimits = None
    defaultCentralParam = None

    def __init_subclass__(cls, **kwargs):
        if not inspect.isabstract(cls):
            for required in (
                "paramDim",
                "dataDim",
            ):
                if not getattr(cls, required):
                    raise AttributeError(
                        f"Can't instantiate abstract class {cls.__name__} without {required} attribute defined"
                    )
        return cls

    def __init__(
        self,
        centralParam: np.ndarray = None,
        paramLimits: np.ndarray = None,
        name: str = None,
    ) -> None:
        # Define model-specific lower and upper limits for the sampling
        # to avoid parameter regions where the evaluation of the model is instable.
        # The limits in the format np.array([lower_dim1, upper_dim1], [lower_dim2, upper_dim2], ...)

        # Define a model-specific central parameter point, which will be used as starting point for the mcmc sampler.
        # A single parameter point in the format np.array([p_dim1, p_dim2, ...])

        # Returns the name of the class to which the object belongs. Overwrite it if you want to
        # give your model a custom name, e. g. depending on the name of your parameters.

        # :return: The class name of the calling object.

        assert centralParam is not None or self.defaultCentralParam is not None
        assert (
            paramLimits is not None
            or self.defaultParamSamplingLimits is not None
        )

        self.centralParam = (
            centralParam if centralParam else self.defaultCentralParam
        )
        self.paramLimits = (
            paramLimits if paramLimits else self.defaultParamSamplingLimits
        )

        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

    @abstractmethod
    def forward(self, param: np.ndarray):
        """Executed the forward pass of the model to obtain data from a parameter. You can also do equivalently :code:`model(param)`.

        Args:
          param: The parameter(set) for which the model should be evaluated.
          param: np.ndarray:

        Returns:

        Raises:
          NotImplementedError: Implement this method to make you model callable.

        """
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, param: np.ndarray):
        """Evaluates the jacobian of the :func:`~epic.core.model.Model.forward` method.

        Args:
          param(np.ndarray): The parameter(set) for which the jacobian of your model should be evaluated.
          param: np.ndarray:

        Returns:
          np.ndarray: The jacobian for the variables returned by the :func:`~epic.core.model.Model.forward` method with respect to the parameters.

        """
        raise NotImplementedError

    def valjac(self, param: np.ndarray):
        """Evaluates the jacobian and the forward pass of the model at the same time. If the method is not overwritten in a subclass it,
        it simply calls :func:`~epic.core.model.Model.forward` and :func:`~epic.core.model.Model.jacobian`.

        Args:
          param(np.ndarray): The parameter(set) for which the model and the jacobian should be evaluated.
          param: np.ndarray:

        Returns:

        """

        return self.forward(param), self.jacobian(param)

    def isArtificial(self) -> bool:
        """Determines whether the model provides artificial data

        Args:

        Returns:
          bool: True if the model inherits from the ArtificialModelInterface

        """
        return issubclass(self.__class__, ArtificialModelInterface)


class ArtificialModelInterface(ABC):
    """By inheriting from this interface you indicate that you are providing an artificial parameter dataset,
    and the corresponding artificial data dataset, which can be used to compare the results from epi with the ground truth.
    The comparison can be done using the plotEmceeResults.

    Args:

    Returns:

    Raises:
      NotImplementedError: Implement the generateArtificialData function to implement this interface.

    """

    @abstractmethod
    def generateArtificialParams(self, numSamples: int):
        """This method must be overwritten an return an numpy array of numSamples parameters.

        Args:
          numSamples: int:

        Returns:

        Raises:
          NotImplementedError: _description_

        """
        raise NotImplementedError

    def generateArtificialData(
        self,
        params: typing.Union[os.PathLike, str, np.ndarray],
    ) -> None:
        """This method is called when the user wants to generate artificial data from the model.

        Args:
          numSamples(int): The number of samples to generate.
          params(np.ndarray or Path or str): The parameters for which the artificial data should be generated.
          params: typing.Union[os.PathLike:
          str:
          np.ndarray]:

        Returns:

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
        try:
            artificialData = vmap(self.forward, in_axes=0)(params)
        except Exception:
            # use standard numpy vectorization if forward is not implemented with jax
            # data = np.vectorize(self.forward)(params)
            artificialData = np.zeros((params.shape[0], 3))
            for i in range(params.shape[0]):
                artificialData[i, :] = self.forward(params[i, :])

        return artificialData


def autodiff(_cls):
    """

    Args:
      _cls:

    Returns:

    """
    _cls.initFwAndBw()
    return _cls


class JaxModel(Model):
    """The JaxModel class automatically creates the jacobian method based on the forward method.
    Additionally it jit compiles the forward and jacobian method with jax.
    To use this class you have to implement your forward method using jax, e. g. jax.numpy.
    Dont overwrite the __init__ method of JaxModel without calling the super constructor.
    Else your forward method wont be jitted.

    Args:
      Model(Model: Model): Abstract parent class

    Returns:

    """

    def __init__(self, name: str = None) -> None:
        super().__init__(name=name)
        # TODO: Check performance implications of not setting this at the class level but for each instance.
        type(self).forward = partial(JaxModel.forward_method, self)

    def __init_subclass__(cls, **kwargs):
        return autodiff(super().__init_subclass__(**kwargs))

    @classmethod
    def initFwAndBw(cls):
        """ """
        # Calculate jitted methods for the subclass(es)
        # It is an unintended sideeffect that this happens for all intermediate classes also.
        # E.g. for: class CoronaArtificial(Corona)
        cls.fw = jit(cls.forward)
        cls.bw = jit(jacrev(cls.forward))
        cls.vj = jit(value_and_jacrev(cls.forward))

    @staticmethod
    def forward_method(self, param):
        """

        Args:
          param:

        Returns:

        """
        return type(self).fw(param)

    def jacobian(self, param):
        """

        Args:
          param:

        Returns:

        """
        return type(self).bw(param)

    def valjac(self, param):
        """

        Args:
          param:

        Returns:

        """
        return type(self).vj(param)
