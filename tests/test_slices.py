"""
The code which is executed here should contain code which represents the typical usage of the epi library.
It could be extended to somehow measure the performance of the code for all examples in a reproducible and comparable way.
At the moment the simplest way to get an indication for the performance is to run the tests and look at the time it takes to run the tests.
For profiling / identifying bottlenecks and not just observing the iterations per seconds, run this file with the following command:
scalene tests/profiling.py from the root directory of the project.
"""
import numpy as np

from epi.core.inference import InferenceType, inference
from epi.core.model import Model
from epi.examples.corona import CoronaArtificial


def test_slices():
    model: Model = CoronaArtificial()

    # generate artificial data
    if model.isArtificial():
        nDataPoints = 1000
        params = model.generateArtificialParams(nDataPoints)
        data = model.generateArtificialData(params)
    else:
        raise Exception("This test is only for artificial data")

    # run MCMC sampling for EPI
    slice1 = np.array([0])
    slice2 = np.array([1, 2])
    inference(model, data, InferenceType.MCMC, slices=[slice1, slice2])
