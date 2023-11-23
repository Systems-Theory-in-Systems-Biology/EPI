"""
Test the model testing function on the Temperature and Heat model.
This especially comprises one model with one 1D input and one 1D output and one function call without default arguments.
"""

from model_test import test_model

from eulerpi.examples.heat import Heat
from eulerpi.examples.temperature import Temperature


def test_model_test():
    """ """

    test_model(Temperature())
    test_model(Heat(), 100, 5000)
