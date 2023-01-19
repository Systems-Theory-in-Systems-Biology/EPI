import epic.plotting.plotter as plotter
from epic.example_models.applications.temperature import TemperatureArtificial


def test_plotKDEoverGrid():
    t = TemperatureArtificial()
    t.generateArtificialData()
    pDim, dDim, numDataPoints, centralParam, data, dataStdevs = t.dataLoader()
    plotter.plotKDEoverGrid(data, dataStdevs, resolution=100)
