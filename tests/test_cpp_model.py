"""Test the cpp model and its equivalent python implementation. Can be used to compare the performance during the sampling."""

import glob
import os
from typing import Type

import pytest

from epic.core.model import Model

# from epic.core.plots import plotTest
from epic.core.sampling import (
    concatenateEmceeSamplingResults,
    runEmceeSampling,
)
from epic.example_models.cpp.cpp_plant import CppPlant
from epic.example_models.cpp.python_reference_plants import (
    ExternalPlant,
    JaxPlant,
)


# TODO: The import of CppPlant can already fail. How to do this elegantly?
# These three models all implement the same "physical model". Therefore i grouped them together. Can be used to compare speed of different approaches
def PlantModels():
    for ModelClass in [CppPlant, JaxPlant, ExternalPlant]:
        if ModelClass == CppPlant:
            yield pytest.param(
                ModelClass,
                marks=pytest.mark.xfail(
                    True, reason="Cpp Library probably not compiled yet"
                ),
            )
        else:
            yield ModelClass


@pytest.mark.xfail(True, reason="Cpp Library probably not compiled yet")
def test_cpp_lib_exists():
    cpp_lib_pattern = "epic/example_models/cpp/cpp_model*.so*"
    file_exists = (
        len([n for n in glob.glob(cpp_lib_pattern) if os.path.isfile(n)]) > 0
    )
    assert file_exists


@pytest.mark.parametrize("ModelClass", PlantModels())
def test_application_model(ModelClass: Type[Model]):
    # define the model
    # TODO: Work in protected directory, not the one where the user probably works
    model: Model = ModelClass(delete=True, create=True)

    # generate artificial data
    if model.isArtificial():
        model.generateArtificialData()

    numWalkers = 10

    # run MCMC sampling for EPI
    runEmceeSampling(model, numWalkers=numWalkers)

    # combine all intermediate saves to create one large sample chain
    concatenateEmceeSamplingResults(model)

    # TODO: Write plotting routine working for all models
    # plot the results
    # plotTest(model)
