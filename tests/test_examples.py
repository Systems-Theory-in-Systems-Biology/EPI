"""
Test the instantiation, data loading / data generation and the inference for the example models in the examples folder.
"""

import importlib

import pytest

from epipy.core.inference import InferenceType, inference
from epipy.core.model import Model

cpp_plant_example = pytest.param(
    ("epipy.examples.cpp.cpp_plant", "CppPlant"),
    marks=pytest.mark.xfail(
        True,
        reason="XFAIL means that the Cpp Library for the plant model ist not compiled yet",
    ),
)


def Examples():
    """Provides the list of examples to the parametrized test"""
    for example in [
        ("epipy.examples.stock", "Stock", "ETF50.csv"),
        ("epipy.examples.stock", "StockArtificial"),
        ("epipy.examples.corona", "Corona", "CoronaData.csv"),
        ("epipy.examples.corona", "CoronaArtificial"),
        ("epipy.examples.temperature", "Temperature", "TemperatureData.csv"),
        ("epipy.examples.temperature", "TemperatureArtificial"),
        (
            "epipy.examples.temperature",
            "TemperatureWithFixedParams",
            "TemperatureData.csv",
        ),
        cpp_plant_example,
        ("epipy.examples.cpp.python_reference_plants", "ExternalPlant"),
        ("epipy.examples.cpp.python_reference_plants", "JaxPlant"),
        ("epipy.examples.sbml.sbml_menten_model", "MentenSBMLModel"),
        ("epipy.examples.sbml.sbml_caffeine_model", "CaffeineSBMLModel"),
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

    # generate artificial data if necessary
    if model.is_artificial():
        num_data_points = 100
        params = model.generate_artificial_params(num_data_points)
        data = model.generate_artificial_data(params)
    else:
        assert dataFileName is not None
        data = importlib.resources.path(module_location, dataFileName)

        if (
            className == "Stock"
        ):  # We check using string comparison because we dont want to statically import the Stock class
            data, _, _ = model.download_data(
                data
            )  # Download the actual stock data from the ticker list data from the internet

    # Run inference
    num_steps = 100
    num_walkers = 12  # We choose 12 because then we have enough walkers for all examples. The higher the dimensionality of the model, the more walkers are needed.
    inference(
        model,
        data,
        inference_type=InferenceType.MCMC,
        num_walkers=num_walkers,
        num_steps=num_steps,
    )

    # TODO: Check if results are correct / models invertible by comparing them with the artificial data for the artificial models
