"""Test pickling of the various classes in the package."""

import pickle

import numpy as np
import pytest

from tests.test_examples import Examples, get_example_name

from .helper import get_model_and_data


@pytest.mark.parametrize("example", Examples(), ids=get_example_name)
def test_pickling_model(example):
    model, _ = get_model_and_data(example)

    # Test pickling
    dumped = pickle.dumps(model)
    loaded = pickle.loads(dumped)

    for key in model.__dict__.keys():
        if key == "amici_model" or key == "amici_solver":
            # Can't compare the models directly, because I dont know how to compare the SwigPyObject objects
            # Iterate over dict and compare each value, besides the SwigPyObject amici_model and amici_solver
            continue
        if isinstance(model.__dict__[key], np.ndarray):
            # catch numpy arrays, they dont implement __eq__ like normal python objects
            assert np.array_equal(model.__dict__[key], loaded.__dict__[key])
        else:
            assert model.__dict__[key] == loaded.__dict__[key]
