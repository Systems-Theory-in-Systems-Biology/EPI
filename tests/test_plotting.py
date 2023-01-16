import epic.core.plotting as plotting
from epic.example_models.applications.temperature import TemperatureArtificial


def test_plotKDEoverGrid():
    t = TemperatureArtificial()
    t.generateArtificialData()
    pDim, dDim, numDataPoints, centralParam, data, dataStdevs = t.dataLoader()
    plotting.plotKDEoverGrid(data, dataStdevs, resolution=100)
