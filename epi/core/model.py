import inspect
import os
from abc import ABC, abstractmethod
from functools import partial
import typing

import numpy as np
from jax import jacrev, jit, vmap

from epi.jax_extension import value_and_jacrev


class Model(ABC):
    """The base class for all models using the EPI algorithm.

    It contains three abstract methods which need to be implemented by subclasses
    """

    paramDim = None
    dataDim = None

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
        self, centralParam: np.ndarray, paramLimits: np.ndarray, name: None) -> None:
        # Define model-specific lower and upper limits for the sampling
        # to avoid parameter regions where the evaluation of the model is instable.
        # The limits in the format np.array([lower_dim1, upper_dim1], [lower_dim2, upper_dim2], ...)

        # Define a model-specific central parameter point, which will be used as starting point for the mcmc sampler.
        # A single parameter point in the format np.array([p_dim1, p_dim2, ...])

        # Returns the name of the class to which the object belongs. Overwrite it if you want to
        # give your model a custom name, e. g. depending on the name of your parameters.

        # :return: The class name of the calling object.

        self.centralParam = centralParam
        self.paramLimits = paramLimits
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

    @abstractmethod
    def forward(self, param: np.ndarray):
        """Executed the forward pass of the model to obtain data from a parameter. You can also do equivalently :code:`model(param)`.

        :param param: The parameter(set) for which the model should be evaluated.
        :raises NotImplementedError: Implement this method to make you model callable.
        """
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, param: np.ndarray):
        """Evaluates the jacobian of the :func:`~epic.core.model.Model.forward` method.

        :param param: The parameter(set) for which the jacobian of your model should be evaluated.
        :type param: np.ndarray
        :return: The jacobian for the variables returned by the :func:`~epic.core.model.Model.forward` method with respect to the parameters.
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def valjac(self, param: np.ndarray):
        """Evaluates the jacobian and the forward pass of the model at the same time. If the method is not overwritten in a subclass it,
        it simply calls :func:`~epic.core.model.Model.forward` and :func:`~epic.core.model.Model.jacobian`.

        :param param: The parameter(set) for which the model and the jacobian should be evaluated.
        :type param: np.ndarray
        """

        return self.forward(param), self.jacobian(param)

    def generateArtificialData(
        self,
        params: typing.Union[os.PathLike, str, np.ndarray],
        numSamples: int,
    ) -> None:
        """This method is called when the user wants to generate artificial data from the model.
        :param numSamples: The number of samples to generate.
        :type numSamples: int
        :param params: The parameters for which the artificial data should be generated.
        :type params: np.ndarray or Path or str
        """
        if isinstance(params, str) or isinstance(params, os.PathLike):
            params = np.loadtxt(params, delimiter=",", ndmin=2)
        elif isinstance(params, np.ndarray):
            pass
        else:
            raise TypeError(
                f"The params argument has to be either a path to a file or a numpy array. The passed argument was of type {type(params)}"
            )
        
        # try to use jax vmap to perform the forward pass on multiple parameters at once
        try:
            artificialData = vmap(self.forward, in_axes=0)(params)
        except:
            # use standard numpy vectorization if forward is not implemented with jax
            #data = np.vectorize(self.forward)(params)
            artificialData = np.zeros((params.shape[0], 3))
            for i in range(params.shape[0]):
                artificialData[i, :] = self.forward(params[i, :])

        return artificialData
            

def autodiff(_cls):
    _cls.initFwAndBw()
    return _cls


class JaxModel(Model):
    """The JaxModel class automatically creates the jacobian method based on the forward method.
    Additionally it jit compiles the forward and jacobian method with jax.
    To use this class you have to implement your forward method using jax, e. g. jax.numpy.
    Dont overwrite the __init__ method of JaxModel without calling the super constructor.
    Else your forward method wont be jitted.

    :param Model: Abstract parent class
    :type Model: Model
    """

    def __init__(self, delete: bool = False, create: bool = True) -> None:
        super().__init__(delete, create)
        # TODO: Check performance implications of not setting this at the class level but for each instance.
        type(self).forward = partial(JaxModel.forward_method, self)

    def __init_subclass__(cls, **kwargs):
        return autodiff(super().__init_subclass__(**kwargs))

    @classmethod
    def initFwAndBw(cls):
        # Calculate jitted methods for the subclass(es)
        # It is an unintended sideeffect that this happens for all intermediate classes also.
        # E.g. for: class CoronaArtificial(Corona)
        cls.fw = jit(cls.forward)
        cls.bw = jit(jacrev(cls.forward))
        cls.vj = jit(value_and_jacrev(cls.forward))

    @staticmethod
    def forward_method(self, param):
        return type(self).fw(param)

    def jacobian(self, param):
        return type(self).bw(param)

    def valjac(self, param):
        return type(self).vj(param)
