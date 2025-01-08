from enum import Enum


class InferenceType(Enum):
    """Available modes for the :py:func:`inference <eulerpi.inference.inference>` function."""

    GRID = 0  #: The grid inference uses a grid to evaluate the joint distribution.
    SAMPLING = 1  #: The SAMPLING / MCMC inference uses a Markov Chain Monte Carlo sampler to sample from the joint distribution.
