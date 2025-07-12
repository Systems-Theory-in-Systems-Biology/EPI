import importlib
import importlib.resources
from typing import Optional, Tuple

from eulerpi.core.models import BaseModel

# Manual cache dictionary keyed by (module_location, class_name)
# We cache the sbml model, because it takes long to compile, and recompilation of the same model
# causes issues because swigs notices this and throws an error, because the module has the same name
# as before and can not be reloaded.
_sbml_model_cache = {}


def is_sbml_model_class(cls):
    return any(base.__name__ == "SBMLModel" for base in cls.__mro__)


def get_model_and_data(example: tuple) -> Tuple[BaseModel, Optional[str]]:
    module_location, class_name, *rest = example
    data_file = rest[0] if rest else None

    module = importlib.import_module(module_location)
    ModelClass = getattr(module, class_name)

    if is_sbml_model_class(ModelClass):
        cache_key = (module_location, class_name)
        if cache_key not in _sbml_model_cache:
            _sbml_model_cache[cache_key] = ModelClass()
        model = _sbml_model_cache[cache_key]
    else:
        model = ModelClass()

    data_resource = None
    if data_file is not None:
        data_resource = importlib.resources.files(module_location).joinpath(
            data_file
        )

    return model, data_resource
