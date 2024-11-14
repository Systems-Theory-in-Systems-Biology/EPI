"""
Test the slices functionality for each of the inference methods.
"""

import importlib

import pytest

from eulerpi.core.inference import inference
from eulerpi.core.models import BaseModel
from eulerpi.examples.temperature import TemperatureWithFixedParams


def temp_name(low_T, high_T):
    # Replace the minus sign with a letter to avoid problems with the file name
    if low_T < 0:
        lTs = str(low_T).replace("-", "m")
    else:
        lTs = str(low_T)
    if high_T < 0:
        hTs = str(high_T).replace("-", "m")
    else:
        hTs = str(high_T)
    name = "TemperatureWithFixedParams_" + lTs + "_" + hTs
    return name


@pytest.mark.parametrize(
    "temperatures",
    [
        (-10.0, 10.0),  # 0
        (-20, 40.0),  # 1
    ],
    ids=[0, 1],
)
def test_fixed_params(temperatures):
    """ """
    model: BaseModel = TemperatureWithFixedParams(*temperatures)
    run_name = temp_name(*temperatures)
    data_resource = importlib.resources.files(
        "eulerpi.examples.temperature"
    ).joinpath("TemperatureData.csv")
    with importlib.resources.as_file(data_resource) as data_file:
        inference(
            model,
            data_file,
            num_steps=100,
            run_name=run_name,
        )
