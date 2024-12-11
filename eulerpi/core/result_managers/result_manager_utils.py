import os

import numpy as np

from eulerpi.core.models.base_model import BaseModel


def get_slice_name(slice: np.ndarray) -> str:
    """This organization function returns a string name for a given slice.

    Args:
        slice(np.ndarray): The slice for which the name will be returned.

    Returns:
        str: The string name of the slice.

    """

    return "slice_" + "".join(["Q" + str(i) for i in slice])


def get_output_path(model: BaseModel) -> str:
    """Returns the path to the output folder, containing also intermediate results.

    Returns:
        str: The path to the output folder, containing also intermediate results.

    """
    return os.path.join("Output", model.name)


def get_run_path(model_name: str, run_name: str) -> str:
    """Returns the path to the folder where the results for the given run are stored.

    Returns:
        str: The path to the folder where the results for the given run are stored.

    """
    return os.path.join("Output", model_name, run_name)
