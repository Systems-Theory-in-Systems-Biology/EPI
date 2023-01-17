from typing import Type

import pytest

from epic.core.model import Model

# from epic.core.plots import plotTest
from epic.core.sampling import (
    concatenateEmceeSamplingResults,
    runEmceeSampling,
)

# TODO: Find better naming or structure for example_models and tests
from epic.example_models.cpp.cpp_plant import CppPlant
from epic.example_models.cpp.python_reference_plants import (
    ExternalPlant,
    JaxPlant,
)


# These three models all implement the same "physical model". Therefore i grouped them together. Can be used to compare speed of different approaches
def PlantModels():
    for ModelClass in [CppPlant, JaxPlant, ExternalPlant]:
        yield ModelClass


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
