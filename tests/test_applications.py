from typing import Type

import pytest

from epic.core.model import Model

# from epic.core.plots import plotTest
from epic.core.sampling import (
    concatenateEmceeSamplingResults,
    runEmceeSampling,
)
from epic.example_models.applications.corona import Corona, CoronaArtificial
from epic.example_models.applications.stock import Stock, StockArtificial
from epic.example_models.applications.temperature import (
    Temperature,
    TemperatureArtificial,
)


def Applications():
    for App in [
        Stock,
        StockArtificial,
        Corona,
        CoronaArtificial,
        Temperature,
        TemperatureArtificial,
    ]:
        yield App


@pytest.mark.parametrize("ModelClass", Applications())
def test_application_model(ModelClass: Type[Model]):
    # define the model
    # TODO: Work in protected directory, not the one where the user probably works
    # Then we can also set delete=True for the tests
    model: Model = ModelClass(delete=True, create=True)

    # generate artificial data
    if model.isArtificial():
        model.generateArtificialData()

    # warnings.simplefilter('always', UserWarning)
    # Avoid RuntimeError: It is unadvisable to use a red-blue move with fewer walkers than twice the number of dimensions
    if issubclass(ModelClass, Stock):
        numWalkers = 12
    else:
        numWalkers = 10

    # run MCMC sampling for EPI
    runEmceeSampling(model, numWalkers=numWalkers)

    # combine all intermediate saves to create one large sample chain
    concatenateEmceeSamplingResults(model)

    # TODO: Write plotting routine working for all models
    # plot the results
    # plotTest(model)
