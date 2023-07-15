import inspect
import os
import tempfile
from abc import ABC, abstractmethod
from functools import partial
from typing import Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit, vmap

import amici
from eulerpi.jax_extension import value_and_jacrev


class Model(ABC):
    """The base class for all models using the EPI algorithm.

    Args:
        central_param(np.ndarray): The central parameter for the model. (Default value = None)
        param_limits(np.ndarray): Box limits for the parameters. The limits are given as a 2D array with shape (param_dim, 2). The parameter limits are used as limits as well as for the movement policy for MCMC sampling, and as boundaries for the grid when using grid-based inference. Overwrite the function param_is_within_domain if the domain is more complex than a box - the grid will still be build based on param_limits, but actual model evaluations only take place within the limits specified in param_is_within_domain. (Default value = None)
        name(str): The name of the model. The class name is used if no name is given. (Default value = None)
    """

    param_dim: Optional[
        np.ndarray
    ] = None  #: The dimension of the parameter space of the model. It must be defined in the subclass.
    data_dim: Optional[
        np.ndarray
    ] = None  #: The dimension of the data space of the model. It must be defined in the subclass.

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
        """Executed the forward pass of the model to obtain data from a parameter. You can also do equivalently :code:`model(param)`.

        Args:
            param(np.ndarray): The parameter for which the data should be generated.

        Returns:
            np.ndarray: The data generated from the parameter.

        """
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, param: np.ndarray) -> np.ndarray:
        """Evaluates the jacobian of the :func:`~eulerpi.core.model.Model.forward` method.

        Args:
            param(np.ndarray): The parameter for which the jacobian should be evaluated.

        Returns:
            np.ndarray: The jacobian for the variables returned by the :func:`~eulerpi.core.model.Model.forward` method with respect to the parameters.

        """
        raise NotImplementedError

    def forward_and_jacobian(
        self, param: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the jacobian and the forward pass of the model at the same time. If the method is not overwritten in a subclass it,
        it simply calls :func:`~eulerpi.core.model.Model.forward` and :func:`~eulerpi.core.model.Model.jacobian`.

        Args:
            param(np.ndarray): The parameter for which the jacobian should be evaluated.

        Returns:
            typing.Tuple[np.ndarray, np.ndarray]: The data generated from the parameter and the jacobian for the variables returned by the :func:`~eulerpi.core.model.Model.forward` method with respect to the parameters.

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

    def is_artificial(self) -> bool:
        """Determines whether the model provides artificial parameter and data sets.

        Returns:
            bool: True if the model inherits from the ArtificialModelInterface

        """
        return issubclass(self.__class__, ArtificialModelInterface)


class ArtificialModelInterface(ABC):
    """By inheriting from this interface you indicate that you are providing an artificial parameter dataset,
    and the corresponding artificial data dataset, which can be used to compare the results from eulerpi with the ground truth.
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
        params: Union[os.PathLike, str, np.ndarray],
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


class JaxModel(Model):
    """The JaxModel class automatically creates the jacobian method based on the forward method.
    Additionally it jit compiles the forward and jacobian method with jax.
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
        """Constructor of the JaxModel class.

        Args:
            name: str: The name of the model. If None the name of the class is used.
        """
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
            np.ndarray: The jacobian for the variables returned by the :func:`~eulerpi.core.model.Model.forward` method with respect to the parameters.

        """
        return type(self).bw(param)

    def forward_and_jacobian(
        self, param: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the jacobian and the forward pass of the model at the same time. This can be more efficient than calling the :func:`~eulerpi.core.model.Model.forward` and :func:`~eulerpi.core.model.Model.jacobian` methods separately.

        Args:
            param(np.ndarray): The parameter for which the jacobian should be evaluated.

        Returns:
            typing.Tuple[np.ndarray, np.ndarray]: The data and the jacobian for a given parameter.

        """
        return type(self).vj(param)


class SBMLModel(Model):
    """The SBMLModel class is a wrapper for the AMICI python interface to simulate SBML models using this package.

    Args:
        sbml_file(str): The path to the SBML model file.
        param_ids(list): A list of ids of parameter, which will be estimated during the inference. If None all parameter ids are extracted from the SBML model.
        state_ids(list): A list of state ids, for which data will be given during the inference. If None all state ids are extracted from the SBML model.
        timepoints(list): List of measurement time points, this is where the sbml model is evaluated and compared to the data
        skip_creation(bool): If True the model is not created againg based on the SBML file. Instead the model is loaded from a previously created model. (Default value = False)
        central_param(np.ndarray): The central parameter for the model
        param_limits(np.ndarray): The parameter limits for the model
    """

    @staticmethod
    def indices_from_ids(ids: list, all_ids: list) -> list:
        """Returns the indices of the ids in the all_ids list.

        Args:
            ids(list): The ids for which the indices should be returned.
            all_ids(list): The list of all ids.

        Returns:
            list: The indices of the ids in the all_ids list.

        Throws:
            ValueError: If one of the ids is not in the all_ids list.

        """
        indices = []
        for id in ids:
            try:
                indices.append(all_ids.index(id))
            except ValueError:
                raise ValueError(
                    f"Parameter / State id '{id}' is not in the list of the relevant ids {all_ids}"
                )
        return indices

    @property
    def param_dim(self):
        """The number of parameters of the model."""
        return len(self.param_ids)

    @property
    def data_dim(self):
        """The dimension of a data point returned by the model."""
        return len(self.state_ids) * len(self.timepoints)

    def __init__(
        self,
        sbml_file: str,
        timepoints: list,
        central_param: np.ndarray,
        param_limits: np.ndarray,
        param_ids: Optional[list] = None,
        state_ids: Optional[list] = None,
        skip_creation: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(central_param, param_limits, name, **kwargs)

        self.amici_model_name = self.name
        self.amici_model_dir = "./amici/" + self.amici_model_name

        # Generate python code
        if not skip_creation:
            sbml_importer = amici.SbmlImporter(sbml_file)
            sbml_importer.sbml2amici(
                self.amici_model_name,
                self.amici_model_dir,
            )

        # Load the generated model
        self.timepoints = timepoints
        self.load_amici_model_and_solver()

        self.param_ids = param_ids or self.amici_model.getParametersIds()
        self.state_ids = state_ids or self.amici_model.getStateIds()
        self.param_indices = self.indices_from_ids(
            self.param_ids, self.amici_model.getParameterIds()
        )
        self.state_indices = self.indices_from_ids(
            self.state_ids, self.amici_model.getStateIds()
        )
        self.setSensitivities()

    def load_amici_model_and_solver(self):
        """Loads the AMICI model from the previously generated model."""
        amici_model_module = amici.import_model_module(
            self.amici_model_name, self.amici_model_dir
        )
        self.amici_model = amici_model_module.getModel()
        self.amici_solver = self.amici_model.getSolver()

        self.amici_model.setTimepoints(self.timepoints)
        self.amici_solver.setAbsoluteTolerance(1e-10)

    def setSensitivities(self):
        if self.param_ids == self.amici_model.getParameterIds():
            self.amici_model.requireSensitivitiesForAllParameters()
        else:
            self.amici_model.setParameterList(self.param_indices)

        self.amici_solver.setSensitivityMethod(amici.SensitivityMethod.forward)
        self.amici_solver.setSensitivityOrder(amici.SensitivityOrder.first)

    def forward(self, params):
        for i, param in enumerate(params):
            self.amici_model.setParameterById(self.param_ids[i], param)
        rdata = amici.runAmiciSimulation(self.amici_model, self.amici_solver)
        return rdata.x[:, self.state_indices].reshape(self.data_dim)

    def jacobian(self, params):
        for i, param in enumerate(params):
            self.amici_model.setParameterById(self.param_ids[i], param)
        rdata = amici.runAmiciSimulation(self.amici_model, self.amici_solver)
        return (
            rdata.sx[:, :, self.state_indices]
            .transpose(1, 0, 2)
            .reshape(self.param_dim, self.data_dim)
            .T
        )

    def forward_and_jacobian(
        self, params: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        for i, param in enumerate(params):
            self.amici_model.setParameterById(self.param_ids[i], param)
        rdata = amici.runAmiciSimulation(self.amici_model, self.amici_solver)
        return (
            rdata.x[:, self.state_indices].reshape(self.data_dim),
            rdata.sx[:, :, self.state_indices]
            .transpose(1, 0, 2)
            .reshape(self.param_dim, self.data_dim)
            .T,
        )

    # Allow the model to be pickled
    def __getstate__(self):
        # Create a copy of the object's state
        state = self.__dict__.copy()

        # Save the amici solver settings to
        _fd, _file = tempfile.mkstemp()

        try:
            # write amici solver settings to file
            try:
                amici.writeSolverSettingsToHDF5(self.amici_solver, _file)
            except AttributeError as e:
                e.args += (
                    "Pickling the SBMLModel requires an AMICI "
                    "installation with HDF5 support.",
                )
                raise
            # read in byte stream
            with open(_fd, "rb", closefd=False) as f:
                state["amici_solver_settings"] = f.read()
        finally:
            # close file descriptor and remove temporary file
            os.close(_fd)
            os.remove(_file)

        state["amici_model_settings"] = amici.get_model_settings(
            self.amici_model
        )

        # Remove the unpicklable entries.
        del state["amici_model"]
        del state["amici_solver"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # Restore amici model and solver
        self.load_amici_model_and_solver()
        self.setSensitivities()

        _fd, _file = tempfile.mkstemp()
        try:
            # write solver settings to temporary file
            with open(_fd, "wb", closefd=False) as f:
                f.write(state["amici_solver_settings"])
            # read in solver settings
            try:
                amici.readSolverSettingsFromHDF5(_file, self.amici_solver)
            except AttributeError as err:
                if not err.args:
                    err.args = ("",)
                err.args += (
                    "Unpickling an AmiciObjective requires an AMICI "
                    "installation with HDF5 support.",
                )
                raise
        finally:
            # close file descriptor and remove temporary file
            os.close(_fd)
            os.remove(_file)

        amici.set_model_settings(
            self.amici_model,
            state["amici_model_settings"],
        )
