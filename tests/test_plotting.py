import epi.plotting.plotter as plotter
from epi.examples.temperature import TemperatureArtificial


def test_plotKDEoverGrid():
    t = TemperatureArtificial()
    t.generateArtificialData()
    pDim, dDim, numDataPoints, centralParam, data, dataStdevs = t.dataLoader()
    plotter.plotKDEoverGrid(data, dataStdevs, resolution=100)
