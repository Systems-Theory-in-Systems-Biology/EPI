import os
import tempfile
from typing import Optional, Tuple

import numpy as np

from .base_model import BaseModel

amici_available = False
try:
    import amici

    amici_available = True
except ImportError:
    pass


def is_amici_available():
    return amici_available


class SBMLModel(BaseModel):
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
        central_param: np.ndarray,
        param_limits: np.ndarray,
        timepoints: list,
        param_ids: Optional[list] = None,
        state_ids: Optional[list] = None,
        skip_creation: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not is_amici_available():
            raise ImportError(
                "The SBMLModel class requires an AMICI installation. "
                "Please install AMICI first."
                "You can do this manually or install eulerpi with the [amici] extra: pip install eulerpi[amici]"
            )

        super().__init__(central_param, param_limits, name, **kwargs)

        self.amici_model_name = self.name
        self.amici_model_dir = (
            "./generated_sbml_models/" + self.amici_model_name
        )

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
        """Tell the underlying amici solver to calculate sensitivities based on the attribute `self.param_ids`"""
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
