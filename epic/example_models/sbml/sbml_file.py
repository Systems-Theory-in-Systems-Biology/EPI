import numpy as np
import sbmltoodepy.modelclasses
from scipy.integrate import odeint


class SBMLmodel(sbmltoodepy.modelclasses.Model):
    def __init__(self):

        self.p = {}  # Dictionary of model parameters

        self.c = {}  # Dictionary of compartments
        self.c["comp"] = sbmltoodepy.modelclasses.Compartment(
            1e-14, 3, True, metadata=sbmltoodepy.modelclasses.SBMLMetadata("")
        )

        self.s = {}  # Dictionary of chemical species
        self.s["E"] = sbmltoodepy.modelclasses.Species(
            5e-21,
            "Amount",
            self.c["comp"],
            False,
            constant=False,
            metadata=sbmltoodepy.modelclasses.SBMLMetadata(""),
        )
        self.s["S"] = sbmltoodepy.modelclasses.Species(
            1e-20,
            "Amount",
            self.c["comp"],
            False,
            constant=False,
            metadata=sbmltoodepy.modelclasses.SBMLMetadata(""),
        )
        self.s["P"] = sbmltoodepy.modelclasses.Species(
            0.0,
            "Amount",
            self.c["comp"],
            False,
            constant=False,
            metadata=sbmltoodepy.modelclasses.SBMLMetadata(""),
        )
        self.s["ES"] = sbmltoodepy.modelclasses.Species(
            0.0,
            "Amount",
            self.c["comp"],
            False,
            constant=False,
            metadata=sbmltoodepy.modelclasses.SBMLMetadata(""),
        )

        self.r = {}  # Dictionary of reactions
        self.r["veq"] = veq(self)
        self.r["vcat"] = vcat(self)

        self.f = {}  # Dictionary of function definitions
        self.time = 0

        self.AssignmentRules()

    def AssignmentRules(self):

        return

    def _SolveReactions(self, y, t):

        self.time = t
        (
            self.s["E"].amount,
            self.s["S"].amount,
            self.s["P"].amount,
            self.s["ES"].amount,
        ) = y
        self.AssignmentRules()

        rateRuleVector = np.array([0, 0, 0, 0], dtype=np.float64)

        stoichiometricMatrix = np.array(
            [[-1, 1.0], [-1, 0.0], [0, 1.0], [1, -1.0]], dtype=np.float64
        )

        reactionVelocities = np.array(
            [self.r["veq"](), self.r["vcat"]()], dtype=np.float64
        )

        rateOfSpeciesChange = (
            stoichiometricMatrix @ reactionVelocities + rateRuleVector
        )

        return rateOfSpeciesChange

    def RunSimulation(
        self, deltaT, absoluteTolerance=1e-12, relativeTolerance=1e-6
    ):

        finalTime = self.time + deltaT
        y0 = np.array(
            [
                self.s["E"].amount,
                self.s["S"].amount,
                self.s["P"].amount,
                self.s["ES"].amount,
            ],
            dtype=np.float64,
        )
        (
            self.s["E"].amount,
            self.s["S"].amount,
            self.s["P"].amount,
            self.s["ES"].amount,
        ) = odeint(
            self._SolveReactions,
            y0,
            [self.time, finalTime],
            atol=absoluteTolerance,
            rtol=relativeTolerance,
            mxstep=5000000,
        )[
            -1
        ]
        self.time = finalTime
        self.AssignmentRules()


class veq:
    def __init__(self, parent, metadata=None):

        self.parent = parent
        self.p = {}
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = sbmltoodepy.modelclasses.SBMLMetadata("")
        self.p["kon"] = sbmltoodepy.modelclasses.Parameter(1000000.0, "kon")
        self.p["koff"] = sbmltoodepy.modelclasses.Parameter(0.2, "koff")

    def __call__(self):
        return self.parent.c["comp"].size * (
            self.p["kon"].value
            * self.parent.s["E"].concentration
            * self.parent.s["S"].concentration
            - self.p["koff"].value * self.parent.s["ES"].concentration
        )


class vcat:
    def __init__(self, parent, metadata=None):

        self.parent = parent
        self.p = {}
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = sbmltoodepy.modelclasses.SBMLMetadata("")
        self.p["kcat"] = sbmltoodepy.modelclasses.Parameter(0.1, "kcat")

    def __call__(self):
        return (
            self.parent.c["comp"].size
            * self.p["kcat"].value
            * self.parent.s["ES"].concentration
        )
