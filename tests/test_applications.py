from typing import Type

import pytest

from epi.core.model import Model

# from epi.core.plots import plotTest
from epi.core.sampling import inference
from epi.examples.corona import Corona, CoronaArtificial
from epi.examples.stock import Stock, StockArtificial
from epi.examples.temperature import Temperature, TemperatureArtificial


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
    model: Model = ModelClass(
        delete=True, create=True
    )  # Delete old results and recreate folder structure

    # generate artificial data
    if model.isArtificial():
        model.generateArtificialData()

    # run MCMC sampling for EPI
    if issubclass(ModelClass, Stock):
        # Avoid RuntimeError: It is unadvisable to use a red-blue move with fewer walkers than twice the number of dimensions
        numWalkers = 12
        inference(model=model, numWalkers=numWalkers)
    else:
        inference(model=model)
