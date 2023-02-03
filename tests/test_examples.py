"""
Test the instantiation, data loading / data generation and the inference for the example models in the examples folder.
"""

import importlib

import pytest

from epi.core.model import Model
from epi.core.sampling import NUM_WALKERS, inference

cpp_plant_example = pytest.param(
    ("epi.examples.cpp", "CppPlant"),
    marks=pytest.mark.xfail(
        True,
        reason="XFAIL means that the Cpp Library for the plant model ist not compiled yet",
    ),
)


def Examples():
    """Provides the list of examples to the parametrized test"""
    for example in [
        ("epi.examples.stock", "Stock", 12),
        ("epi.examples.stock", "StockArtificial", 12),
        # ("epi.examples.corona", "Corona"),
        ("epi.examples.corona", "CoronaArtificial"),
        ("epi.examples.temperature", "Temperature"),
        ("epi.examples.temperature", "TemperatureArtificial"),
        ("epi.examples.temperature", "TemperatureWithFixedParams"),
        cpp_plant_example,
        ("epi.examples.cpp", "ExternalPlant"),
        ("epi.examples.cpp", "JaxPlant"),
    ]:
        yield example


def getExampleName(example):
    """Extract the name of the example from the tuple to have nice names in the test report and be able to select the test using -k"""
    return example[1]


@pytest.mark.parametrize("example", Examples(), ids=getExampleName)
def test_examples(example):
    """

    :param example: The example which should be tested. The tuple contains the module location, the class name and the number of walkers.
    :type example: Tuple[str, str, int]
    """
    # extract example parameters from tuple
    try:
        module_location, className, numWalkers = example
    except ValueError:
        module_location, className = example
        numWalkers = NUM_WALKERS

    # Import class dynamically to avoid error on imports at the top which cant be tracked back to a specific test
    module = importlib.import_module(module_location)
    ModelClass = getattr(module, className)

    model: Model = ModelClass(
        delete=True, create=True
    )  # Delete old results and recreate folder structure

    # generate artificial data if necessary
    if model.isArtificial():
        model.generateArtificialData()

    inference(
        model=model, numWalkers=numWalkers, numSteps=1000
    )  # using default dataPath of the model
