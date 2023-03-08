"""
Test the slices functionality for each of the inference methods.
"""
import numpy as np
import pytest

from epi.core.inference import InferenceType, inference
from epi.core.model import Model
from epi.examples.stock import StockArtificial

# Parametrize the test to run for each inference type


@pytest.mark.parametrize(
    "inference_type",
    InferenceType._member_map_.values(),
    ids=InferenceType._member_names_,
)
def test_slices(inference_type):
    """ """
    model: Model = StockArtificial()

    # generate artificial data
    if model.isArtificial():
        nDataPoints = 1000
        params = model.generateArtificialParams(nDataPoints)
        data = model.generateArtificialData(params)
    else:
        raise Exception("This test is only for artificial data")

    slice1 = np.array([0])
    slice2 = np.array([1, 2])
    slice3 = np.array([3, 4, 5])
    slices = [slice1, slice2, slice3]

    inference(model, data, inference_type, slices=slices)
