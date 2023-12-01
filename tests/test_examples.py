"""
Test the instantiation, data loading / data generation and the inference for the example models in the examples folder.
"""

import importlib

import pytest

from eulerpi.core.inference import InferenceType, inference
from eulerpi.core.model import Model, is_amici_available
from eulerpi.core.model_check import basic_model_check

cpp_plant_example = pytest.param(
    ("eulerpi.examples.cpp.cpp_plant", "CppPlant"),
    marks=pytest.mark.xfail(
        True,
        reason="XFAIL means that the Cpp Library for the plant model ist not compiled yet",
    ),
)

menten_example = pytest.param(
    ("eulerpi.examples.sbml.sbml_menten_model", "MentenSBMLModel"),
    marks=pytest.mark.skipif(
        not is_amici_available(),
        reason="Amici not available, no SBML models can be tested",
    ),
)

caffeine_example = pytest.param(
    ("eulerpi.examples.sbml.sbml_caffeine_model", "CaffeineSBMLModel"),
    marks=pytest.mark.skipif(
        not is_amici_available(),
        reason="Amici not available, no SBML models can be tested",
    ),
)


def Examples():
    """Provides the list of examples to the parametrized test"""
    for example in [
        ("eulerpi.examples.heat", "HeatArtificial"),
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
        menten_example,
        caffeine_example,
    ]:
        yield example


def get_example_name(example):
    """Extract the name of the example from the tuple to have nice names in the test report and be able to select the test using -k

    Args:
      example: Tuple of the form (module_location, className, dataFileName) or (module_location, className)

    Returns:
        str: name of the example

    """
    return example[1]


@pytest.mark.parametrize("example", Examples(), ids=get_example_name)
def test_model_requirements(example):
    """Perform a simple sanity check on the model.

    Args:
        example: Tuple of the form (module_location, className, dataFileName) or (module_location, className)
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

    # Use the basic model check to assert all model attribute dimension requirements as well as forward simulations and the corresponding jacobian.
    basic_model_check(model)


@pytest.mark.parametrize("example", Examples(), ids=get_example_name)
@pytest.mark.parametrize("inference_type", list(InferenceType))
def test_examples(example, inference_type):
    """Test the inference for the example models.

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

    params, sim_res, densities, result_manager = inference(
        model,
        data,
        **kwargs,
    )

    # get all keys of the params dict
    full_slice_str = list(params.keys())[0]

    assert result_manager is not None

    assert params.keys() == sim_res.keys() == densities.keys()
    assert params[full_slice_str].shape[0] == sim_res[full_slice_str].shape[0]
    assert (
        params[full_slice_str].shape[0] == densities[full_slice_str].shape[0]
    )

    assert params[full_slice_str].shape[1] == model.param_dim
    assert (
        sim_res[full_slice_str].shape[1] == model.data_dim
    )  # Take care, only valid for full slice

    # TODO: Check if results are correct / models invertible by comparing them with the artificial data for the artificial models
