import epic.core.plotting as plotting
from epic.example_models.applications.temperature import TemperatureArtificial

t = TemperatureArtificial()
pDim, dDim, numDataPoints, centralParam, data, dataStdevs = t.dataLoader()


def test_plotKDEoverGrid():
    plotting.plotKDEoverGrid(data, dataStdevs, resolution=100)
