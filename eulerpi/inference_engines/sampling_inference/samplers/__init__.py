"""This subpackage defines a sampler base class and an implementation to be used in the sampling based inference"""

from .emcee_sampler import EmceeSampler
from .sampler import Sampler

__all__ = [
    "EmceeSampler",
    "Sampler",
]
