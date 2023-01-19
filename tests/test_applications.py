from typing import Type

import pytest

from epi.core.model import Model

# from epi.core.plots import plotTest
from epi.core.sampling import concatenateEmceeSamplingResults, runEmceeSampling
from epi.example_models.applications.corona import Corona, CoronaArtificial
from epi.example_models.applications.stock import Stock, StockArtificial
from epi.example_models.applications.temperature import (
    Temperature,
    TemperatureArtificial,
)


def Applications():
    """Provides the list of applications to the parametrized test"""
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
    """Test wether the Model of Type ModelClass can initiate the necessary folder structure and run through the epi algorithm.
    Plotting will maybe also be tested

    :param ModelClass: The Model which will be tested
    :type ModelClass: Type[Model]
    """
    # define the model
    # TODO: Work in protected directory, not the one where the user probably works
    # Then we can also set delete=True for the tests
    model: Model = ModelClass(delete=True, create=True)

    # generate artificial data
    if model.isArtificial():
        model.generateArtificialData()

    # run MCMC sampling for EPI
    if issubclass(ModelClass, Stock):
        # Avoid RuntimeError: It is unadvisable to use a red-blue move with fewer walkers than twice the number of dimensions
        numWalkers = 12
        runEmceeSampling(model, numWalkers=numWalkers)
    else:
        runEmceeSampling(model)

    # combine all intermediate saves to create one large sample chain
    concatenateEmceeSamplingResults(model)
