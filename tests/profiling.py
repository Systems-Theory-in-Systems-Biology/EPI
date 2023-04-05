"""
The code which is executed here should contain code which represents the typical usage of the epipy library.
It could be extended to somehow measure the performance of the code for all examples in a reproducible and comparable way.
At the moment the simplest way to get an indication for the performance is to run the tests and look at the time it takes to run the tests.
For profiling / identifying bottlenecks and not just observing the iterations per seconds, run this file with the following command:
scalene tests/profiling.py from the root directory of the project.
"""

import numpy as np

from epipy.core.inference import InferenceType, inference
from epipy.core.model import Model
from epipy.examples.stock import StockArtificial


def profiling_with_slices(inference_type):
    """ """
    model: Model = StockArtificial()

    # generate artificial data
    num_data_points = 1000
    params = model.generate_artificial_params(num_data_points)
    data = model.generate_artificial_data(params)

    slice1 = np.array([0])
    slice2 = np.array([1, 2])
    slice3 = np.array([3, 4, 5])
    slices = [slice1, slice2, slice3]
    inference(
        model,
        data,
        inference_type,
        slices=slices,
    )


if __name__ == "__main__":
    profiling_with_slices(InferenceType.MCMC)
    profiling_with_slices(InferenceType.DENSE_GRID)
    profiling_with_slices(InferenceType.SPARSE_GRID)
