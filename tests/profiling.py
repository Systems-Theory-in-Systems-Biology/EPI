"""
The code which is executed here should contain code which represents the typical usage of the eulerpi library.
It could be extended to somehow measure the performance of the code for all examples in a reproducible and comparable way.
At the moment the simplest way to get an indication for the performance is to run the tests and look at the time it takes to run the tests.
For profiling / identifying bottlenecks and not just observing the iterations per seconds, run this file with the following command:
scalene tests/profiling.py from the root directory of the project.
"""

import jax
import numpy as np

from eulerpi.examples.corona import CoronaArtificial
from eulerpi.inference import InferenceType, inference
from eulerpi.models import BaseModel


def profiling_with_slices(inference_type):
    """ """
    model: BaseModel = CoronaArtificial()

    # generate artificial data
    num_data_points = 1000
    params = model.generate_artificial_params(num_data_points)
    data = model.generate_artificial_data(params)

    slice1 = np.array([0])
    slice2 = np.array([1, 2])
    slice3 = np.array([0, 1, 2])
    slices = [slice1, slice2, slice3]
    for slice in slices:
        inference(
            model,
            data,
            inference_type,
            slice=slice,
            num_steps=5000,
        )


if __name__ == "__main__":
    with jax.log_compiles():
        profiling_with_slices(InferenceType.SAMPLING)
        profiling_with_slices(InferenceType.GRID)
