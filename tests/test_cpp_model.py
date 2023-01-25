"""Test the cpp model and its equivalent python implementation. Can be used to compare the performance during the sampling."""

import glob
import os
from typing import Type

import pytest

from epi.core.model import Model
from epi.core.sampling import inference
from epi.examples.cpp import CppPlant, ExternalPlant, JaxPlant


# TODO: The import of CppPlant can already fail. How to do this elegantly?
# These three models all implement the same "physical model". Therefore i grouped them together. Can be used to compare speed of different approaches
def PlantModels():
    for ModelClass in [CppPlant, JaxPlant, ExternalPlant]:
        # The CppPlant is not available if the cpp library is not compiled yet
        # Therefore we have to return this model class together with a pytest mark that this test may fail
        if ModelClass == CppPlant:
            yield pytest.param(
                ModelClass,
                marks=pytest.mark.xfail(
                    True,
                    reason="XFAIL means that the Cpp Library for the plant model ist not compiled yet",
                ),
            )
        else:
            yield ModelClass


@pytest.mark.xfail(
    True,
    reason="XFAIL means that the Cpp Library for the plant model ist not compiled yet",
)
def test_cpp_lib_exists():
    cpp_lib_pattern = "epi/examples/cpp/cpp_model*.so*"
    file_exists = (
        len([n for n in glob.glob(cpp_lib_pattern) if os.path.isfile(n)]) > 0
    )
    assert file_exists


@pytest.mark.parametrize("ModelClass", PlantModels())
def test_application_model(ModelClass: Type[Model]):
    model: Model = ModelClass(
        delete=True, create=True
    )  # Delete old results and recreate folder structure

    # generate artificial data
    if model.isArtificial():
        model.generateArtificialData()

    inference(model=model)
