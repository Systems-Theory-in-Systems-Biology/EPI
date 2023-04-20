"""
Test the instantiation, data loading / data generation and the inference for the example models in the examples folder.
"""

import importlib

import pytest

from eulerpi.core.inference import InferenceType, inference
from eulerpi.core.model import Model

cpp_plant_example = pytest.param(
    ("eulerpi.examples.cpp.cpp_plant", "CppPlant"),
    marks=pytest.mark.xfail(
        True,
        reason="XFAIL means that the Cpp Library for the plant model ist not compiled yet",
    ),
)


def Examples():
    """Provides the list of examples to the parametrized test"""
    for example in [
        ("eulerpi.examples.stock", "Stock", "ETF50.csv"),
        ("eulerpi.examples.stock", "StockArtificial"),
        ("eulerpi.examples.corona", "Corona", "CoronaData.csv"),
        ("eulerpi.examples.corona", "CoronaArtificial"),
        ("eulerpi.examples.temperature", "Temperature", "TemperatureData.csv"),
        ("eulerpi.examples.temperature", "TemperatureArtificial"),
        (
            "eulerpi.examples.temperature",
            "TemperatureWithFixedParams",
            "TemperatureData.csv",
        ),
        cpp_plant_example,
        ("eulerpi.examples.cpp.python_reference_plants", "ExternalPlant"),
        ("eulerpi.examples.cpp.python_reference_plants", "JaxPlant"),
        ("eulerpi.examples.sbml.sbml_menten_model", "MentenSBMLModel"),
        ("eulerpi.examples.sbml.sbml_caffeine_model", "CaffeineSBMLModel"),
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
@pytest.mark.parametrize("inference_type", list(InferenceType))
def test_examples(example, inference_type):
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

    # Define kwargs for inference
    kwargs = {}
    kwargs["inference_type"] = inference_type
    if inference_type == InferenceType.MCMC:
        kwargs["num_walkers"] = max(4, model.param_dim * 2)
        kwargs["num_steps"] = 10
    elif inference_type == InferenceType.DENSE_GRID:
        kwargs["num_grid_points"] = 3
    elif inference_type == InferenceType.SPARSE_GRID:
        kwargs["num_levels"] = 3

    inference(
        model,
        data,
        **kwargs,
    )

    # TODO: Check if results are correct / models invertible by comparing them with the artificial data for the artificial models
