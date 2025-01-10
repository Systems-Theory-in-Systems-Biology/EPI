from eulerpi import InferenceResult
from eulerpi.result_managers import PathManager
from pathlib import Path
from typing import Union


def load_inference_result(path: Union[str, Path]) -> InferenceResult:
    pass


def load_inference_result(
    model_name: str, run_name: str, path_manager: PathManager = None
) -> InferenceResult:
    pass
