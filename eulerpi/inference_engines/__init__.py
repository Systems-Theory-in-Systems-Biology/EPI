"""This subpackage defines an abstract base class (`InferenceEngine`) that serves as an interface for solving parameter inference problems.
It also includes concrete implementations of inference engines."""

from .grid_inference import GridInferenceEngine
from .inference_engine import InferenceEngine
from .inference_type import InferenceType
from .inference_result import InferenceResult
from .sampling_inference import SamplingInferenceEngine

__all__ = [
    "GridInferenceEngine",
    "InferenceEngine",
    "InferenceResult",
    "SamplingInferenceEngine",
    "InferenceType",
]
