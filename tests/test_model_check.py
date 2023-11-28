"""
Test the model checking function on the Temperature and Heat model.
This especially comprises one model with one 1D input and one 1D output and one function call without default arguments.
"""

from eulerpi.core.model_check import full_model_check
from eulerpi.examples.heat import Heat
from eulerpi.examples.temperature import Temperature


def test_full_model_check():
    """ """

    full_model_check(Temperature())
    full_model_check(Heat(), 100, 5000)
