"""
Test the instantiation, data loading / data generation and the inference for the example models in the examples folder.
"""

import importlib

import pytest

from epi.core.inference import InferenceType, inference
from epi.core.model import Model

cpp_plant_example = pytest.param(
    ("epi.examples.cpp.cpp_plant", "CppPlant"),
    marks=pytest.mark.xfail(
        True,
        reason="XFAIL means that the Cpp Library for the plant model ist not compiled yet",
    ),
)


def Examples():
    """Provides the list of examples to the parametrized test"""
    for example in [
        ("epi.examples.stock", "Stock", "ETF50.csv"),
        ("epi.examples.stock", "StockArtificial"),
        ("epi.examples.corona", "Corona", "CoronaData.csv"),
        ("epi.examples.corona", "CoronaArtificial"),
        ("epi.examples.temperature", "Temperature", "TemperatureData.csv"),
        ("epi.examples.temperature", "TemperatureArtificial"),
        (
            "epi.examples.temperature",
            "TemperatureWithFixedParams",
            "TemperatureData.csv",
        ),
        cpp_plant_example,
        ("epi.examples.cpp.python_reference_plants", "ExternalPlant"),
        ("epi.examples.cpp.python_reference_plants", "JaxPlant"),
    ]:
        yield example


def get_example_name(example):
    """Extract the name of the example from the tuple to have nice names in the test report and be able to select the test using -k

    Args:
      example:

    Returns:

    """
    return example[1]


@pytest.mark.parametrize("example", Examples(), ids=get_example_name)
def test_examples(example):
    """

    Args:
      example(Tuple[str, str, int]): The example which should be tested. The tuple contains the module location, the class name and the number of walkers.

    Returns:

    """
    # extract example parameters from tuple
    try:
        module_location, className, dataFileName = example
    except ValueError:
        module_location, className = example
        dataFileName = None

    # Import class dynamically to avoid error on imports at the top which cant be tracked back to a specific test
    module = importlib.import_module(module_location)
    ModelClass = getattr(module, className)
    model: Model = ModelClass()
    # TODO: Delete old results and recreate folder structure

    # generate artificial data if necessary
    if model.is_artificial():
        num_data_points = 1000
        params = model.generate_artificial_params(num_data_points)
        data = model.generate_artificial_data(params)
    else:
        assert dataFileName is not None
        data = importlib.resources.path(module_location, dataFileName)

        if (
            className == "Stock"
        ):  # We check using string comparison because we dont want to statically import the Corona class
            data, _, _ = model.download_data(
                data
            )  # Download the actual stock data from the ticker list data from the internet

    # Run inference
    num_steps = 1000
    num_walkers = 12  # We choose 12 because then we have enough walkers for all examples. The higher the dimensionality of the model, the more walkers are needed.
    inference(
        model,
        data,
        inference_type=InferenceType.MCMC,
        num_walkers=num_walkers,
        num_steps=num_steps,
    )

    # TODO: Check if results are correct by comparing them with the artificial data for the artificial models
