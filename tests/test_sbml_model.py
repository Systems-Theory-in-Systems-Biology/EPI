import pytest

# from epi.core.plots import plotTest
from epi.core.sampling import concatenateEmceeSamplingResults, runEmceeSampling
from epi.examples.sbml import MySBMLModel


# TODO
@pytest.mark.skip(reason="Not Correctly implemented yet")
def test_sbml_model():
    # define the model
    model = MySBMLModel(filepath="epi/examples/sbml/sbml_file.xml")

    # generate artificial data
    model.generateArtificialData()

    # run MCMC sampling for EPI
    runEmceeSampling(model)

    # combine all intermediate saves to create one large sample chain
    concatenateEmceeSamplingResults(model)
