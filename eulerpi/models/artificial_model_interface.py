import os
from abc import ABC, abstractmethod
from typing import Union

import jax.numpy as jnp
import numpy as np


class ArtificialModelInterface(ABC):
    """By inheriting from this interface you indicate that you are providing an artificial parameter dataset,
    and the corresponding artificial data dataset, which can be used to compare the results from eulerpi with the ground truth.

    """

    @abstractmethod
    def generate_artificial_params(self, num_samples: int) -> np.ndarray:
        """This method must be overwritten an return an numpy array of num_samples parameters.

        Args:
            num_samples(int): The number of parameters to generate.

        Returns:
            np.ndarray: The generated parameters.

        Raises:
            NotImplementedError: If the method is not overwritten in a subclass.

        """
        raise NotImplementedError

    def generate_artificial_data(
        self,
        params: Union[os.PathLike, str, np.ndarray],
    ) -> np.ndarray:
        """This method is called when the user wants to generate artificial data from the model.

        Args:
            params: typing.Union[os.PathLike, str, np.ndarray]: The parameters for which the data should be generated. Can be either a path to a file, a numpy array or a string.

        Returns:
            np.ndarray: The data generated from the parameters.

        Raises:
            TypeError: If the params argument is not a path to a file, a numpy array or a string.

        """
        if isinstance(params, str) or isinstance(params, os.PathLike):
            params = np.loadtxt(params, delimiter=",", ndmin=2)
        elif isinstance(params, np.ndarray) or isinstance(params, jnp.ndarray):
            pass
        else:
            raise TypeError(
                f"The params argument has to be either a path to a file or a numpy array. The passed argument was of type {type(params)}"
            )

        return self.forward_vectorized(params)
