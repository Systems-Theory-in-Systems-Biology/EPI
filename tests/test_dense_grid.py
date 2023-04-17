"""
Test the slices functionality for each of the inference methods.
"""
import numpy as np
import pytest

from eulerpi.core.dense_grid_types import DenseGridType
from eulerpi.core.inference import inference
from eulerpi.core.inference_types import InferenceType
from eulerpi.core.model import Model
from eulerpi.examples.stock import StockArtificial


# Parametrize the test to run for each inference type
@pytest.mark.parametrize(
    "dense_grid_type",
    list(DenseGridType),
    ids=DenseGridType._member_names_,
)
def test_dense_grid(dense_grid_type):
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
    inference(
        model,
        data,
        inference_type=InferenceType.DENSE_GRID,
        slices=slices,
        dense_grid_type=dense_grid_type,
        num_grid_points=4,
    )
