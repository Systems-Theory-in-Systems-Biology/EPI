"""Test plotting functions from the plotter module"""

import epi.plotting.plotter as plotter
from epi.core.kde import calcKernelWidth
from epi.examples.temperature import TemperatureArtificial


def test_plotKDEoverGrid():
    """Test plotting a KDE over a grid."""
    t = TemperatureArtificial()
    params = t.generateArtificialParams()
    data = t.generateArtificialData(params)
    dataStdevs = calcKernelWidth(data)
    plotter.plotKDEoverGrid(data, dataStdevs, resolution=100)
