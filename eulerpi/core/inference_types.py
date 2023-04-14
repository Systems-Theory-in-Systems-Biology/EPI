from enum import Enum


# Define an enum for the inference types: DenseGrid, MCMC
class InferenceType(Enum):
    """The type of inference to be used."""

    DENSE_GRID = 0  #: The dense grid inference uses a dense grid to evaluate the joint distribution.
    MCMC = 1  #: The MCMC inference uses a Markov Chain Monte Carlo sampler to sample from the joint distribution.
    SPARSE_GRID = 2  #: The sparse grid inference uses a sparse grid to evaluate the joint distribution. It is not tested and not recommended.
