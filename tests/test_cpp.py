"""
Test the cpp model and its equivalent python implementation. Can be used to compare the performance during the sampling.
"""

import importlib

import pytest


@pytest.mark.xfail(
    True,
    reason="XFAIL means that the Cpp Library for the plant model ist not compiled yet",
)
def test_cpp_lib_exists():
    """Test if the cpp library for the plant model exists."""
    cpp_path = importlib.resources.path("epi.examples.cpp", "")

    # Check if the cpp library exists by checking if an .so file exists in the cpp directory
    assert any(
        [file.suffix == ".so" for file in cpp_path.iterdir()]
    ), "No cpp library found in the cpp directory"
