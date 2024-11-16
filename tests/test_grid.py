"""
Test the slices functionality for each of the inference methods.
"""

import numpy as np
import pytest

from eulerpi.epi import InferenceType, inference
from eulerpi.examples.corona import CoronaArtificial
from eulerpi.grids.grid_factory import GRID_REGISTRY
from eulerpi.models import ArtificialModelInterface, BaseModel


# Parametrize the test to run for each inference type
@pytest.mark.parametrize(
    "grid_type",
    GRID_REGISTRY.keys(),
    ids=GRID_REGISTRY.keys(),
)
def test_grid(grid_type):
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
    inference(
        model,
        data,
        inference_type=InferenceType.GRID,
        slices=slices,
        grid_type=grid_type,
        num_grid_points=4,
    )
