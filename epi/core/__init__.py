from .model import ArtificialModelInterface as ArtificialModelInterface
from .model import JaxModel as JaxModel
from .model import Model as Model
from .model import VisualizationModelInterface as VisualizationModelInterface

# Expose the core functionality of the package as epi.core.xxx
from .sampling import inference as inference
