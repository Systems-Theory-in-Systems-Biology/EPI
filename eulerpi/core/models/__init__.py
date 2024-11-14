"""Package containing an abstract model base class and specialized subclasses for use with the EPI algorithm."""

from .artificial_model_interface import (
    ArtificialModelInterface as ArtificialModelInterface,
)
from .base_model import BaseModel as BaseModel  # Backwards Compatibility
from .base_model import BaseModel as Model
from .jax_model import JaxModel as JaxModel
from .sbml_model import SBMLModel as SBMLModel

__all__ = [
    "ArtificialModelInterface",
    "BaseModel",
    "JaxModel",
    "SBMLModel",
    "Model",
]
