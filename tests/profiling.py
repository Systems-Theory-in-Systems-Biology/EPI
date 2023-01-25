"""
The code which is executed here should contain code which represents the typical usage of the epi library.
It could be extended to somehow measure the performance of the code for all examples in a reproducible and comparable way.
At the moment the simplest way to get an indication for the performance is to run the tests and look at the time it takes to run the tests.
For profiling / identifying bottlenecks and not just observing the iterations per seconds, run this file with the following command:
scalene tests/profiling.py from the root directory of the project.
"""
from epi.core.model import Model
from epi.core.sampling import inference
from epi.examples.corona import CoronaArtificial

if __name__ == "__main__":
    model: Model = CoronaArtificial()

    # generate artificial data
    if model.isArtificial():
        model.generateArtificialData()

    # run MCMC sampling for EPI
    numWalkers = 12
    inference(model=model, numWalkers=numWalkers)
