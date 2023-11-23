"""
Test the model checking function on the Temperature and Heat model.
This especially comprises one model with one 1D input and one 1D output and one function call without default arguments.
"""

from eulerpi.core.model_check import check_model
from eulerpi.examples.heat import Heat
from eulerpi.examples.temperature import Temperature


def test_model_check():
    """ """

    check_model(Temperature())
    check_model(Heat(), 100, 5000)
