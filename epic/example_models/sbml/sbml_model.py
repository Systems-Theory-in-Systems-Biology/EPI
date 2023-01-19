import copy
import importlib
from enum import Enum
from pathlib import Path

import diffrax as dx
import jax.numpy as jnp
import libsbml as sbml
import numpy as np
import sbmltoodepy
import sbmltoodepy.modelclasses  # Syntax highlighting for sbmltoodepy.modelclasse.Model not working without this import for vs code

# import sys
# import os
# sys.stdout, sys.stderr = os.devnull, os.devnull # silence command-line output temporarily
from pysces import model as PyscesModel
from pysces.PyscesInterfaces import Core2interfaces as PyscesCore2Interface

from epic import logger
from epic.core.model import Model

# from epic.example_models.sbml.sbml_file import SBMLmodel as ExternSBMLModel

# sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__# unsilence command-line output
# Use __SILENT_START__ ?


def check_sbml(filepath: str):
    reader = sbml.SBMLReader()
    document: sbml.SBMLDocument = reader.readSBMLFromFile(filepath)
    return document.getNumErrors() == 0


# file is only file name, path is directory
def toPysces(sbml_file: str, path: str) -> PyscesModel:
    interface = PyscesCore2Interface()
    interface.convertSBML2PSC(
        sbmlfile="sbml_file.xml",
        sbmldir=path,
        pscdir=path,
        pscfile="sbml_file.psc",
    )
    model = PyscesModel(File="sbml_file.psc", dir=path)
    model.showModel()
    model.showODE()
    model.showODEr()
    return model


# file is only file, path is dir
def toOdePy(sbml_file: str, path: str) -> sbmltoodepy.modelclasses.Model:
    sbmltoodepy.ParseAndCreateModel(path + sbml_file)
    module = importlib.import_module(
        "epic.example_models.sbml_model." + sbml_file
    )
    model = module.SBMLmodel()  # todo: set this correctly
    return model


class Converters(Enum):
    OdePy = "OdeOy"
    PySces = "OySces"


class MySBMLModel(Model):
    """The SBML model allows to pass an sbml file, which defines the ODE of a biological system,
    to generate the forward and jacobian of the model automatically.
    """

    def __init__(self, filepath: str, ignore_errors=False) -> None:
        super().__init__()
        p = Path(filepath)
        assert p.is_file()

        if p.suffix == ".xml":
            n_errors = check_sbml(filepath)
            if n_errors > 0 and not ignore_errors:
                raise RuntimeError(
                    "The parsing of the sbml raised errors. You can try to ignore them by passing `ignore_errors=True`"
                )
        elif p.suffix == ".psc":
            raise NotImplementedError(
                "Currently not supporting psc files. Easy to fix ;) Make a PR if urgent."
            )
        else:
            raise ValueError(
                f"File with unsupported file ending passed. Only xml(sbml) files and psc files are supported. The observed suffix is {p.suffix}"
            )

        self.inner_model = toPysces(
            p.name, p.parent
        )  # I dont really like this way of getting filename and directory.

    def forward(self, param):
        # TODO: Wtf
        s0_sim_init = copy.copy(self.inner_model.__inspec__)
        if self.inner_model.__HAS_RATE_RULES__:
            # TODO: Unpacking in ode evaluation to x, vtemp
            s0_sim_init = np.concatenate([s0_sim_init, self.__rrule__])

        print("initial s", s0_sim_init)

        xInit = jnp.array(s0_sim_init)

        # TODO. what to do with param?
        def rhs(t, x, param):
            self.inner_model._EvalODE_CVODE(x)

        term = dx.ODETerm(rhs)
        solver = dx.Kvaerno5()
        # TODO: Times from sbml file? are there any?
        saveat = dx.SaveAt(ts=[0.0, 1.0, 2.0, 5.0, 15.0])
        stepsize_controller = dx.PIDController(rtol=1e-5, atol=1e-5)

        try:
            odeSol = dx.diffeqsolve(
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
            return odeSol.ys[1:5, 2]

        except Exception as e:
            logger.warn("ODE solution not possible!", exc_info=e)
            return np.array([-np.inf, -np.inf, -np.inf, -np.inf])

    # def _SolveReactions(self, y, t):
    #     self.time = t
    #     self.s['E'].amount, self.s['S'].amount, self.s['P'].amount, self.s['ES'].amount = y
    #     self.AssignmentRules()
    #     rateRuleVector = np.array([ 0, 0, 0, 0], dtype = np.float64)
    #     stoichiometricMatrix = np.array([[-1,1.],[-1,0.],[ 0,1.],[ 1,-1.]], dtype = np.float64)
    #     reactionVelocities = np.array([self.r['veq'](), self.r['vcat']()], dtype = np.float64)
    #     rateOfSpeciesChange = stoichiometricMatrix @ reactionVelocities + rateRuleVector
    #     return rateOfSpeciesChange

    # def RunSimulation(self, deltaT, absoluteTolerance = 1e-12, relativeTolerance = 1e-6):
    #     finalTime = self.time + deltaT
    #     y0 = np.array([self.s['E'].amount, self.s['S'].amount, self.s['P'].amount, self.s['ES'].amount], dtype = np.float64)
    #     self.s['E'].amount, self.s['S'].amount, self.s['P'].amount, self.s['ES'].amount = odeint(self._SolveReactions, y0, [self.time, finalTime], atol = absoluteTolerance, rtol = relativeTolerance, mxstep=5000000)[-1]
    #     self.time = finalTime
    #     self.AssignmentRules()

    # TODO: Implement?
    def getCentralParam(self) -> np.ndarray:
        raise NotImplementedError("TODO in sbml model")

    def getParamSamplingLimits(self) -> np.ndarray:
        raise NotImplementedError("TODO in sbml model")
