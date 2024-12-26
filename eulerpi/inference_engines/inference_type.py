from enum import StrEnum


class InferenceType(StrEnum):
    """Available modes for the :py:func:`inference <eulerpi.inference.inference>` function."""

    GRID = "GRID"  #: The grid inference uses a grid to evaluate the joint distribution.
    SAMPLING = "SAMPLING"  #: The SAMPLING / MCMC inference uses a Markov Chain Monte Carlo sampler to sample from the joint distribution.
