"""Test pickling of the various classes in the package."""

import dill as pickle
import numpy as np


def test_pickle_SBML():
    from eulerpi.examples.sbml.sbml_caffeine_model import CaffeineSBMLModel

    model = CaffeineSBMLModel(skip_creation=True)
    dumped = pickle.dumps(model)
    loaded = pickle.loads(dumped)

    # Can't compare the models directly, because I dont know how to compare the SwigPyObject objects
    # Iterate over dict and compare each value, besides the SwigPyObject amici_model and amici_solver
    for key in model.__dict__.keys():
        if key == "amici_model" or key == "amici_solver":
            continue
        # catch numpy arrays
        if isinstance(model.__dict__[key], np.ndarray):
            assert np.array_equal(model.__dict__[key], loaded.__dict__[key])
        else:
            assert model.__dict__[key] == loaded.__dict__[key]
