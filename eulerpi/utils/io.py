import os
import pathlib
from typing import Union

import jax.numpy as jnp
import numpy as np


def load_data(
    data: Union[str, os.PathLike, pathlib.Path, np.ndarray, jnp.ndarray]
) -> np.ndarray:
    if isinstance(data, (str, os.PathLike)):
        data = np.loadtxt(data, delimiter=",", ndmin=2)
    elif not isinstance(data, (np.ndarray, jnp.ndarray)):
        raise TypeError(
            f"The data argument must be a path to a file or a numpy array. The argument passed was of type {type(data)}."
        )
    return data
