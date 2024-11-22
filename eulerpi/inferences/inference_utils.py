import os
import pathlib
from typing import Union

import jax.numpy as jnp
import numpy as np
import psutil


def _load_data(data: Union[str, os.PathLike, pathlib.Path]) -> np.ndarray:
    if isinstance(data, (str, os.PathLike, pathlib.Path)):
        data = np.loadtxt(data, delimiter=",", ndmin=2)
    elif not isinstance(data, (np.ndarray, jnp.ndarray)):
        raise TypeError(
            f"The data argument must be a path to a file or a numpy array. The argument passed was of type {type(data)}."
        )
    return data


def _num_processors_available():
    return psutil.cpu_count(logical=False)
