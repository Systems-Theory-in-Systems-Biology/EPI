import jax
from jax import config

from .inference import InferenceType as InferenceType
from .inference import inference as inference

config.update("jax_enable_x64", True)

# TODO: Remove restriction to cpu
jax.config.update("jax_platform_name", "cpu")  # Restricting to cpu for now
