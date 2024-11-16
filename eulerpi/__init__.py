import jax
from jax import config

from .epi import InferenceType as InferenceType
from .epi import inference as inference

config.update("jax_enable_x64", True)

# TODO: Remove restriction to cpu
jax.config.update("jax_platform_name", "cpu")  # Restricting to cpu for now
