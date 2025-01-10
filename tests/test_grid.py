"""
Test the slices functionality for each of the inference methods.
"""

import importlib

import numpy as np
import pytest

from eulerpi.examples.corona import CoronaArtificial
from eulerpi.inference_engines.grid_inference.grids.grid import Grid
from eulerpi.inference import inference
from eulerpi.inference_engines.inference_type import InferenceType


def Grids():
    """Provides the list of grids for the test function"""
    for example in [
        (
            "eulerpi.grids.equidistant_grid",
            "EquidistantGrid",
            {"num_grid_points": 10},
        ),
        (
            "eulerpi.grids.chebyshev_grid",
            "ChebyshevGrid",
            {"num_grid_points": 10},
        ),
        ("eulerpi.grids.sparse_grid", "SparseGrid", {"max_level_sum": 3}),
    ]:
        yield example


def get_grid_name(example):
    return example[1]


@pytest.mark.parametrize("grid_tuple", Grids(), ids=get_grid_name)
def test_grid(grid_tuple):
    """ """
    model = CoronaArtificial()

    num_data_points = 100
    params = model.generate_artificial_params(num_data_points)
    data = model.generate_artificial_data(params)

    # Import class dynamically to avoid error on imports at the top which cant be tracked back to a specific test
    module_location, className, grid_kwargs = grid_tuple
    module = importlib.import_module(module_location)
    GridClass: type[Grid] = getattr(module, className)

    slice1 = np.array([0])
    slice2 = np.array([1, 2])
    slice3 = np.array([0, 1, 2])
    slices = [slice1, slice2, slice3]
    for slice in slices:
        grid = GridClass(model.param_limits[slice], **grid_kwargs)
        inference(
            model,
            data,
            inference_type=InferenceType.GRID,
            slice=slice,
            grid=grid,
        )
