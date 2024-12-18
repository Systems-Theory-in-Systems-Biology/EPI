"""The inference module provides the main interface to the eulerpi library in the form of the :py:func:`inference <inference>` function."""

import os
import pathlib
from typing import Union

from eulerpi.data_transformations import DataNormalization, DataTransformation
from eulerpi.evaluation import KDE, GaussKDE
from eulerpi.models import BaseModel
from eulerpi.utils.io import load_data


class InferenceProblem:
    def __init__(
        self,
        model: BaseModel,
        data: Union[str, os.PathLike, pathlib.Path],
        data_transformation: DataTransformation = None,
        kde: KDE = None,
    ):
        self.model = model

        self.data = load_data(data)

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
