import jax
from jax import config

from .inference_engines.inference_type import InferenceType as InferenceType
from .inference import inference as inference
from .inference_engines import InferenceResult as InferenceResult
from .load import load_inference_result as load_inference_result

config.update("jax_enable_x64", True)

# TODO: Remove restriction to cpu
jax.config.update("jax_platform_name", "cpu")  # Restricting to cpu for now
