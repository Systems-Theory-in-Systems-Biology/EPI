"""Test the cpp model and its equivalent python implementation. Can be used to compare the performance during the sampling."""

import glob
import os

import pytest


@pytest.mark.xfail(
    True,
    reason="XFAIL means that the Cpp Library for the plant model ist not compiled yet",
)
def test_cpp_lib_exists():
    cpp_lib_pattern = "epi/examples/cpp/cpp_model*.so*"
    file_exists = (
        len([n for n in glob.glob(cpp_lib_pattern) if os.path.isfile(n)]) > 0
    )
    assert file_exists
