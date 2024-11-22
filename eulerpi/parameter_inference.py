"""The inference module provides the main interface to the eulerpi library in the form of the :py:func:`inference <inference>` function."""

import os
import pathlib
from typing import Optional, Tuple, Union

import numpy as np

from eulerpi.data_transformations import DataNormalization, DataTransformation
from eulerpi.evaluation import KDE, GaussKDE
from eulerpi.inferences.grid_inference_module import (
    grid_inference as do_grid_inference,
)
from eulerpi.inferences.inference_utils import (
    _load_data,
    _num_processors_available,
)
from eulerpi.inferences.sampling_inference_module import (
    sampling_inference as do_sampling_inference,
)
from eulerpi.models import BaseModel
from eulerpi.result_manager import ResultManager


class ParameterInference:
    def __init__(
        self,
        model: BaseModel,
        data: Union[str, os.PathLike, pathlib.Path],
        data_transformation: DataTransformation = None,
        kde: KDE = None,
    ):
        self.model = model

        self.data = _load_data(data)

        # Assign the DataNormalization as default value
        if data_transformation is None:
            data_transformation = DataNormalization(data)
        if not isinstance(data_transformation, DataTransformation):
            raise TypeError(
                f"The data_transformation must be an instance of a subclass of DataTransformation. It is of type {type(data_transformation)}."
            )
        self.data_transformation = data_transformation
        self.data = data_transformation.transform(self.data)

        if kde is None:
            kde = GaussKDE(data)
        if not isinstance(kde, KDE):
            raise TypeError(
                f"The kde must be an instance of a subclass of KDE. It is of type {type(kde)}."
            )
        self.kde = kde
        self.num_processors_available = _num_processors_available()

    def grid_inference(
        self,
        slice=None,
        result_manager=None,
        num_processes=None,
        grid=None,
        num_grid_points=10,
        load_balancing_safety_factor=1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ResultManager]:
        if slice is None:
            slice = np.arange(
                self.model.param_dim
            )  # If no slice is given, compute full joint distribution, i.e. a slice with all parameters        num_processes = num_processes or self.num_processes_available
        do_grid_inference(
            self.model,
            self.data_transformation,
            self.kde,
            slice,
            result_manager,
            num_processes,
            grid,
            num_grid_points,
            load_balancing_safety_factor,
        )

    def sampling_inference(
        self,
        slice=None,
        result_manager=None,
        num_processes=None,
        sampler=None,
        num_walkers: int = 10,
        num_steps: int = 2500,
        num_burn_in_samples: Optional[int] = None,
        thinning_factor: Optional[int] = None,
        get_walker_acceptance=False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ResultManager]:
        if slice is None:
            slice = np.arange(
                self.model.param_dim
            )  # If no slice is given, compute full joint distribution, i.e. a slice with all parameters        num_processes = num_processes or self.num_processes_available
        do_sampling_inference(
            self.model,
            self.data_transformation,
            self.kde,
            slice,
            result_manager,
            num_processes,
            sampler,
            num_walkers,
            num_steps,
            num_burn_in_samples,
            thinning_factor,
            get_walker_acceptance,
        )
