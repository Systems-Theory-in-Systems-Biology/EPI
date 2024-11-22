from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np


class Sampler(ABC):
    @abstractmethod
    def run(
        self,
        logdensity_blob_function: Callable,
        initial_walker_positions: np.ndarray,
        num_walkers: int,
        num_steps: int,
        num_processes: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "The run method needs to be implemented by every Sampler"
        )
