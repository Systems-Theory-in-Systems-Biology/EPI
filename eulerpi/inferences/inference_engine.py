from abc import ABC, abstractmethod

import numpy as np

from eulerpi.data_transformations.data_transformation import DataTransformation
from eulerpi.evaluation.kde import KDE
from eulerpi.models.base_model import BaseModel
from eulerpi.result_managers import OutputWriter


class InferenceEngine(ABC):
    def __init__(
        self,
        model: BaseModel,
        data_transformation: DataTransformation,
        kde: KDE,
    ):
        self.model = model
        self.data_transformation = data_transformation
        self.kde = kde

    @abstractmethod
    def run(
        self,
        slice: np.ndarray,
        output_writer: OutputWriter,
        num_processes: int,
        **kwargs: dict,
    ):
        raise NotImplementedError(
            "The run method must be implemented for all InferenceEngines"
        )
