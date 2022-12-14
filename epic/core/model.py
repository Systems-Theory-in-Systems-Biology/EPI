import abc
import math
from typing import Callable

import numpy as np
from jax import jacrev
from jax.config import config

from epic.core.kernel_density_estimation import calcKernelWidth

config.update("jax_enable_x64", True)

# TODO: Everywhere, how to deal with emcee requiring pickable but jax is not?!


class Model(abc.ABC):
    """The base class for all models using the EPI algorithm.

    It contains three abstract methods which need to be implemented by subclasses
    """

    def __init__(self) -> None:
        # self.modelName = ""
        self.jacrev_jacobian = None
        # self.centralParam : np.array = None

        # self.paramBounds : np.array = None #[[lower, upper], ...]
        # self.dataBounds : np.array = None  #[[lower, upper], ...]
        # TODO: Determine if saving this is usefull and when/how calc it
        # self.dataDim for visgrid
        # self.paramDim for ""

        # TODO: Also need this for each model....
        # self.paramsLowerLimits = None
        # self.paramsUpperLimits = None
        pass

    def __call__(self, param):
        return self.forward(param)

    # Define model-specific lower and upper borders for sampling
    # to avoid parameter regions where the simulation can only be evaluated instably.
    @abc.abstractmethod
    def getParamSamplingLimits(self) -> np.ndarray:
        raise NotImplementedError

    def getModelName(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def getCentralParam(self) -> np.ndarray:
        raise NotImplementedError

    def dataLoader(
        self,
    ) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
        paramDim = self.getCentralParam().shape[0]

        data = np.loadtxt(
            "Data/" + self.getModelName() + "Data.csv", delimiter=","
        )
        if len(data.shape) == 1:
            data = data.reshape((data.shape[0], 1))

        dataStdevs = calcKernelWidth(data)
        numDataPoints, dataDim = data.shape

        return (
            paramDim,
            dataDim,
            numDataPoints,
            self.getCentralParam(),
            data,
            dataStdevs,
        )

    def getForward(self) -> Callable:
        return self.forward

    # TODO: Best practice avilable for this?
    def getJacobian(self) -> Callable:
        self.jacrev_jacobian = jacrev(self.forward)
        return self.jacrev_jacobian

    @abc.abstractmethod
    def forward(self, param):
        """Executed the forward pass of the model to obtain data from a parameter. Do not call the forward method! Instead do :code:`model(param)`.

        :param param: The parameter(set) for which the model should be evaluated.
        :raises NotImplementedError: If the method is not implemented in the subclass
        """
        raise NotImplementedError

    def jacobian(self, param):
        """Evaluates the jacobian of the :func:`~epic.core.model.Model.forward` method. If the method is not provided in the subclass
        the jacobian is calculated by  :func:`jax.jacrev`.

        :param param: _description_
        :type param: _type_
        :return: _description_
        :rtype: _type_
        """
        if self.jacrev_jacobian is None:
            self.jacrev_jacobian = jacrev(self.forward)
        return self.jacrev_jacobian(param)

    def correction(self, param) -> np.double:
        """Evaluate the pseudo-determinant of the simulation jacobian (that serves as a correction term) in one specific parameter point.

        :parameter param: parameter at which the simulation model is evaluated
        :return: correction correction factor for density transformation)
        """

        # Evaluate the algorithmic differentiation object in the parameter
        jac = self.jacobian(param)
        jacT = np.transpose(jac)

        # The pseudo-determinant is calculated as the square root of the determinant of the matrix-product of the Jacobian and its transpose.
        # For numerical reasons, one can regularize the matrix product by adding a diagonal matrix of ones before calculating the determinant.
        # correction = np.sqrt(np.linalg.det(np.matmul(jacT,jac) + np.eye(param.shape[0])))
        correction = np.sqrt(np.linalg.det(np.matmul(jacT, jac)))

        # If the correction factor is not a number or infinite, return 0 instead to not affect the sampling.
        if math.isnan(correction) or math.isinf(correction):
            correction = 0.0
            print("invalid value encountered for correction factor")

        return correction


# TODO: Its not really an interface?
# TODO: Inherit from Model or Interface? (See python-interface)
class ArtificialModelInterface(abc.ABC):
    @abc.abstractmethod
    def generateArtificialData(self) -> None:
        raise NotImplementedError

    def paramLoader(self) -> tuple[np.ndarray, np.ndarray]:
        """Load and return all parameters for artificial set ups

        Args:
            model

        Returns:
            _type_: params (true parameters used to generate artificial data)
                    paramStdevs (array of suitable kernel standard deviations for each parameter dimension)
        """
        trueParams = np.loadtxt(
            "Data/" + self.getModelName() + "Params.csv", delimiter=","
        )

        if len(trueParams.shape) == 1:
            trueParams = trueParams.reshape((trueParams.shape[0], 1))

        paramStdevs = calcKernelWidth(trueParams)

        return trueParams, paramStdevs


# TODO: Its not really an interface?
# TODO: Inherit from Model or Interface? (See python-interface)
class VisualizationModelInterface(abc.ABC):
    """Provides the function for the generation of the dataGrid and paramGrid for the visualization of the distributions.
    It forces subclasses to implement the abstract methods  getParamBounds: and getDataBounds:.
    """

    @abc.abstractmethod
    def getParamBounds(self) -> np.ndarray:
        """Returns the bounds on the parameters used to visualize the parameter distribution.

        Returns:
              np.array: An array of the form [[lowerLimit_dim1, upperLimit_dim1], [lowerLimit_dim2, upperLimit_dim2],...]

        Raises:
            NotImplementedError: If the method is not implemented in the subclass
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getDataBounds(self) -> np.ndarray:
        """Returns the bounds on the data used to visualize the data distribution.

        Returns:
              np.array: An array of the form [[lowerLimit_dim1, upperLimit_dim1], [lowerLimit_dim2, upperLimit_dim2],...]

        Raises:
            NotImplementedError: If the method is not implemented in the subclass
        """
        raise NotImplementedError

    def generateVisualizationGrid(
        self, resolution: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # allocate storage for the parameter and data plotting grid
        paramGrid = np.zeros((resolution, self.paramDim))
        dataGrid = np.zeros((resolution, self.dataDim))
        for d in range(self.dataDim):
            dataGrid[:, d] = np.linspace(*self.dataBounds[d, :], resolution)
        for d in range(self.paramDim):
            dataGrid[:, d] = np.linspace(*self.paramBounds[d, :], resolution)

        # store both grids as csv files into the model-specific plot directory
        np.savetxt(
            "Applications/" + self.getModelName() + "/Plots/dataGrid.csv",
            dataGrid,
            delimiter=",",
        )
        np.savetxt(
            "Applications/" + self.getModelName() + "/Plots/paramGrid.csv",
            paramGrid,
            delimiter=",",
        )
        return dataGrid, paramGrid
