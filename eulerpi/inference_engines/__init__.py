"""This subpackage offers an abstract base class/interface for an InferenceEngine, which solves the parameter inference problem, and readily available concrete implementations."""

from .grid_inference_engine import GridInferenceEngine
from .inference_engine import InferenceEngine
from .inference_type import InferenceType
from .sampling_inference_engine import SamplingInferenceEngine
