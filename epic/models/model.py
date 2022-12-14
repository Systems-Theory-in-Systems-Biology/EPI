import abc
import math

import numpy as np
from jax import jacrev
from jax.config import config

from epic.kernel_density_estimation import calcKernelWidth

config.update("jax_enable_x64", True)

# TODO: Everywhere, how to deal with emcee requiring pickable but jax is not?!

class Model(abc.ABC):
    def __init__(self) -> None:
        self.modelName = ""

        self.centralParam : np.array = None

        self.paramBounds : np.array = None #[[lower, upper], ...]
        self.dataBounds : np.array = None  #[[lower, upper], ...]
        # TODO: Determine if saving this is usefull and when/how calc it
        # self.dataDim for visgrid
        # self.paramDim for ""

        # TODO: Also need this for each model....
        # self.paramsLowerLimits = None
        # self.paramsUpperLimits = None
        pass

    def getModelName(self):
        return self.__class__.__name__

    def getCentralParam(self):
        raise NotImplementedError

    def getParamBounds(self):
        raise NotImplementedError
    def getDataBounds(self):
        raise NotImplementedError

    def generateVisualizationGrid(self, resolution):
            # allocate storage for the parameter and data plotting grid
        paramGrid = np.zeros((resolution, self.paramDim))
        dataGrid = np.zeros((resolution, self.dataDim))
        for d in range(self.dataDim):
            dataGrid[:, d] = np.linspace(*self.dataBounds[d,:], resolution)
        for d in range(self.paramDim):
            dataGrid[:, d] = np.linspace(*self.paramBounds[d,:], resolution)
        
        # store both grids as csv files into the model-specific plot directory
        np.savetxt(
            "Applications/" + self.modelName + "/Plots/dataGrid.csv",
            dataGrid,
            delimiter=",",
        )
        np.savetxt(
            "Applications/" + self.modelName + "/Plots/paramGrid.csv",
            paramGrid,
            delimiter=",",
        )
        return dataGrid, paramGrid

    def dataLoader(self):
        paramDim = self.centralParam.shape[0]

        data = np.loadtxt("Data/" + self.modelName + "Data.csv", delimiter=",")
        if len(data.shape) == 1:
            data = data.reshape((data.shape[0], 1))

        dataStdevs = calcKernelWidth(data)
        numDataPoints, dataDim = data.shape

        return paramDim, dataDim, numDataPoints, self.centralParam, data, dataStdevs

    # TODO: The model loader doesn not make sense in the model class
    # Replaced by constructor?
    # def modelLoader():
    #     pass

    def getForward(self):
        return self.forward

    def getJacobian(self):
        return jacrev(self.forward)

    def forward(self, param):
        pass

    # Optional? Else obtain by jacrev(forward) ???
    def jacobian(self, param):
        pass

    def correction(self, param):
        """Evaluate the pseudo-determinant of the simulation jacobian (that serves as a correction term) in one specific parameter point.

        :parameter modelJac: (algorithmic differentiation object for Jacobian of the sim model)
        :parameter param: (parameter at which the simulation model is evaluated)
        :return: correction (correction factor for density transformation)
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

#TODO: Its not really an interface?
#TODO: Inherit from Model or Interface? (See python-interface)
class ArtificialModelInterface(abc.ABC):
    @abc.abstractmethod
    def generateArtificialData(self):
        raise NotImplementedError

    def paramLoader(self):
        """Load and return all parameters for artificial set ups

        Args:
            modelName (_type_): Model ID

        Returns:
            _type_: params (true parameters used to generate artificial data)
                    paramStdevs (array of suitable kernel standard deviations for each parameter dimension)
        """
        trueParams = np.loadtxt("Data/" + self.modelName + "Params.csv", delimiter=",")

        if len(trueParams.shape) == 1:
            trueParams = trueParams.reshape((trueParams.shape[0], 1))

        paramStdevs = calcKernelWidth(trueParams)

        return trueParams, paramStdevs