"""Test plotting functions from the plotter module"""

import epi.plotting.plotter as plotter
from epi.examples.temperature import TemperatureArtificial


def test_plotKDEoverGrid():
    """Test plotting a KDE over a grid."""
    t = TemperatureArtificial()
    t.generateArtificialData()
    dataDim, data, dataStdevs = t.dataLoader()
    plotter.plotKDEoverGrid(data, dataStdevs, resolution=100)
