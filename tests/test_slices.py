"""
Test the slices functionality for each of the inference methods.
"""
import numpy as np
import pytest

from epipy.core.inference import InferenceType, inference
from epipy.core.model import Model
from epipy.examples.stock import StockArtificial

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
    if model.is_artificial():
        num_data_points = 100
        params = model.generate_artificial_params(num_data_points)
        data = model.generate_artificial_data(params)
    else:
        raise Exception("This test is only for artificial data")

    slice1 = np.array([0])
    slice2 = np.array([1, 2])
    slice3 = np.array([3, 4, 5])
    slices = [slice1, slice2, slice3]
    if inference_type == InferenceType.MCMC:
        kwargs = {"num_steps": 100}
    else:
        kwargs = {}
    inference(
        model,
        data,
        inference_type,
        slices=slices,
        **kwargs,
    )
