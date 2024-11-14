import jax
from jax import config

from .models import ArtificialModelInterface as ArtificialModelInterface
from .models import BaseModel as BaseModel
from .models import JaxModel as JaxModel
from .models import SBMLModel as SBMLModel

config.update("jax_enable_x64", True)

# TODO: Remove restriction to cpu
jax.config.update("jax_platform_name", "cpu")  # Restricting to cpu for now
