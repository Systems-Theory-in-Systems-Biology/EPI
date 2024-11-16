"""
Test the slices functionality for each of the inference methods.
"""

import numpy as np
import pytest

from eulerpi.epi import InferenceType, inference
from eulerpi.examples.corona import CoronaArtificial
from eulerpi.models import ArtificialModelInterface, BaseModel

# Parametrize the test to run for each inference type


@pytest.mark.parametrize(
    "inference_type",
    InferenceType._member_map_.values(),
    ids=InferenceType._member_names_,
)
def test_slices(inference_type):
    """ """
    model: BaseModel = CoronaArtificial()

    # generate artificial data
    if isinstance(model, ArtificialModelInterface):
        num_data_points = 100
        params = model.generate_artificial_params(num_data_points)
        data = model.generate_artificial_data(params)
    else:
        raise Exception("This test is only for artificial data")

    slice1 = np.array([0])
    slice2 = np.array([1, 2])
    slice3 = np.array([0, 1, 2])
    slices = [slice1, slice2, slice3]
    if inference_type == InferenceType.SAMPLING:
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
