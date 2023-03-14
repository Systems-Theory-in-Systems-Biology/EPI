"""Test plotting functions from the plotter module"""

import epi.plotting.plotter as plotter
from epi.core.kde import calc_kernel_width
from epi.examples.temperature import TemperatureArtificial


def test_plotKDEoverGrid():
    """Test plotting a KDE over a grid."""
    t = TemperatureArtificial()
    params = t.generate_artificial_params()
    data = t.generate_artificial_data(params)
    data_stdevs = calc_kernel_width(data)
    plotter.plotKDEoverGrid(data, data_stdevs, resolution=100)
