import jax.numpy as jnp
import libsbml as sbml
import numpy as np

from epic.core.model import JaxModel


class MySBMLModel(JaxModel):
    def __init__(self, sbml_file) -> None:
        super().__init__()
        reader = sbml.SBMLReader()
        document: sbml.SBMLDocument = reader.readSBMLFromFile(
            "./epic/example_models/sbml_model/sbml_file.xml"
        )
        assert document.getNumErrors() == 0

        self.sbml_model: sbml.Model = document.getModel()
        # ast = sbml.readMathMLFromString()
        param_idx = 1
        parameter_species = self.sbml_model.getSpecies(param_idx)
        parameter_initial = self.sbml_model.getInitialAssignment(
            param_idx
        )  # ???
        parameter_fw = 0
        for r_i in range(self.sbml_model.getNumReactions()):
            parameter_fw += self.sbml_model.getReaction(r_i)

    def getCentralParam(self) -> np.ndarray:
        return np.array([0.0] * self.sbml_model.getNumSpecies())

    def forward(self, param):
        return jnp.array(param) ** 2
