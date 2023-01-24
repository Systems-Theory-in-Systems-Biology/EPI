import math
import os
import shutil
from abc import ABC, abstractmethod

# import jax.numpy as jnp
import numpy as np
import seedir
from jax import jacrev, jit
from jax.config import config
from seedir import FakeDir, FakeFile

from epi import logger
from epi.core.kde import calcKernelWidth

config.update("jax_enable_x64", True)


class Model(ABC):
    """The base class for all models using the EPI algorithm.

    It contains three abstract methods which need to be implemented by subclasses
    """

    def __init__(self, delete: bool = False, create: bool = True) -> None:
        if delete:
            self.deleteApplicationFolderStructure()
            self.createApplicationFolderStructure()
        elif create:
            self.createApplicationFolderStructure()

        self.setDataPath(
            "Data/" + self.getModelName() + "Data.csv"
        )  # Set default data path

    @abstractmethod
    def getParamSamplingLimits(self) -> np.ndarray:
        """Define model-specific lower and upper limits for the sampling
        to avoid parameter regions where the evaluation of the model is instable.

        :raises NotImplementedError: Implement this method allow the mcmc sampler to work stably
        :return: The limits in the format np.array([lower_dim1, upper_dim1], [lower_dim2, upper_dim2], ...)
        :rtype: np.ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def getCentralParam(self) -> np.ndarray:
        """Define a model-specific central parameter point, which will be used as starting point for the mcmc sampler.

        :raises NotImplementedError: Implement this method to provide a good starting point for the mcmc sampler.
        :return: A single parameter point in the format np.array([p_dim1, p_dim2, ...])
        :rtype: np.ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, param: np.ndarray):
        """Executed the forward pass of the model to obtain data from a parameter. Do not call the forward method! Instead do :code:`model(param)`.

        :param param: The parameter(set) for which the model should be evaluated.
        :raises NotImplementedError: Implement this method to make you model callable.
        """
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, param: np.ndarray):
        """Evaluates the jacobian of the :func:`~epic.core.model.Model.forward` method. If the method is not provided in the subclass
        the jacobian is calculated by  :func:`jax.jacrev`.

        :param param: The parameter(set) for which the jacobian of your model should be evaluated.
        :type param: np.ndarray
        :return: The jacobian for the variables returned by the :func:`~epic.core.model.Model.forward` method with respect to the parameters.
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def correction(self, param: np.ndarray) -> np.double:
        """Evaluate the pseudo-determinant of the simulation jacobian (that serves as a correction term) in one specific parameter point.

        :parameter param: parameter at which the simulation model is evaluated
        :return: correction correction factor for density transformation)
        """

        # Evaluate the algorithmic differentiation object in the parameter
        jac = self.jacobian(param)
        jac = np.atleast_2d(jac)
        jacT = np.transpose(jac)

        # The pseudo-determinant is calculated as the square root of the determinant of the matrix-product of the Jacobian and its transpose.
        # For numerical reasons, one can regularize the matrix product by adding a diagonal matrix of ones before calculating the determinant.
        # correction = np.sqrt(np.linalg.det(np.matmul(jacT,jac) + np.eye(param.shape[0])))
        correction = np.sqrt(np.linalg.det(np.matmul(jacT, jac)))

        # If the correction factor is not a number or infinite, return 0 instead to not affect the sampling.
        if math.isnan(correction) or math.isinf(correction):
            correction = 0.0
            logger.warning("Invalid value encountered for correction factor")
        return correction

    def setDataPath(self, path: str) -> None:
        """Set the path to the data file which shall be used from now on.

        :param path: The path to the data file.
        :type path: str
        """
        self.dataPath = path

    def dataLoader(
        self,
    ) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
        """Load the data from the data file found under the models current data path and calculate several properties of the data.

        :return: The dimension of the parameter space, the dimension of the data space, the number of data points, the central parameter point, the data and the estimated optimal kernel width for each dimension of the data.
        """
        paramDim = self.getCentralParam().shape[0]
        centralParam = self.getCentralParam()

        data = np.loadtxt(self.dataPath, delimiter=",")

        if len(data.shape) == 1:
            data = data.reshape((data.shape[0], 1))

        dataStdevs = calcKernelWidth(data)
        numDataPoints, dataDim = data.shape

        return (
            paramDim,
            dataDim,
            numDataPoints,
            centralParam,
            data,
            dataStdevs,
        )

    def loadSimResults(self, numBurnSamples: int, occurrence: int):
        """Load the files generated by the EPI algorithm through sampling

        :param model: Model from which the results will be loaded
        :type model: Model
        :param numBurnSamples: Ignore the first samples of each chain
        :type numBurnSamples: int
        :param occurrence: step of sampling from chains
        :type occurrence: int
        :return: _description_
        :rtype: _type_
        """
        densityEvals = np.loadtxt(
            self.getApplicationPath() + "/OverallDensityEvals.csv",
            delimiter=",",
        )[numBurnSamples::occurrence]
        simResults = np.loadtxt(
            self.getApplicationPath() + "/OverallSimResults.csv",
            delimiter=",",
        )[numBurnSamples::occurrence, :]
        paramChain = np.loadtxt(
            self.getApplicationPath() + "/OverallParams.csv",
            delimiter=",",
        )[numBurnSamples::occurrence, :]
        return densityEvals, simResults, paramChain

    def getModelName(self) -> str:
        """Returns the name of the class to which the object belongs. Overwrite it if you want to
        give your model a custom name, e. g. depending on the name of your parameters.

        :return: The class name of the calling object.
        :rtype: str
        """
        return self.__class__.__name__

    def getApplicationPath(self) -> str:
        """Returns the path to the simulation results folder, containing also intermediate results

        :return: path as string to the simulation folder
        :rtype: str
        """
        path = "Applications/" + self.getModelName()
        return path

    def createApplicationFolderStructure(self) -> None:
        """Creates the `Application` folder including subfolder where all simulation results
        are stored for this model. No files are deleted during this action.
        """
        indent = 4
        plotFolderTree = (
            (" " * indent + "- SpiderWebs/ \n" + " " * indent + "- Plots/")
            if self.isVisualizable()
            else ""
        )

        os.makedirs("Data", exist_ok=True)
        applicationFolderStructure = (
            "Applications/ \n"
            "  - {modelName}/ \n"
            "    - DensityEvals/ \n"
            "    - Params/ \n"
            "    - SimResults/ \n"
        )
        path = "."
        structure = applicationFolderStructure + plotFolderTree

        def create(f, root):
            fpath = f.get_path()
            joined = os.path.join(root, fpath)
            if isinstance(f, FakeDir):
                try:
                    os.mkdir(joined)
                except FileExistsError:
                    logger.info(f"Directory `{joined}` already exists")
            elif isinstance(f, FakeFile):
                try:
                    with open(joined, "w"):
                        pass
                except FileExistsError:
                    logger.info(f"File `{joined}` already exists")

        fakeStructure = seedir.fakedir_fromstring(
            structure.format(modelName=self.getModelName())
        )
        fakeStructure.realize = lambda path_arg: fakeStructure.walk_apply(
            create, root=path_arg
        )
        fakeStructure.realize(path)

    def deleteApplicationFolderStructure(self):
        """Deletes the models `Applications` subfolder"""
        try:
            shutil.rmtree(self.getApplicationPath())
        except FileNotFoundError:
            logger.info(
                f"Directory `{self.getApplicationPath()}` can't be deleted, "
                "because it does not exist."
            )

    # TODO: Rethink this hidden structure. Seems to be somehow the wrong way round
    # with the underscores.
    def __call__(self, param):
        return self._forward(param)

    def _forward(self, param):
        return self.forward(param)

    def _jacobian(self, param):
        return self.jacobian(param)

    def isArtificial(self) -> bool:
        """Determines whether the model provides artificial data

        :return: True if the model inherits from the ArtificialModelInterface
        :rtype: bool
        """
        return issubclass(self.__class__, ArtificialModelInterface)

    def isVisualizable(self) -> bool:
        """Determines whether the model provides bounds for the visualization grids

        :return: True if the model inherits from the VisualizationModelInterface
        :rtype: bool
        """
        return issubclass(self.__class__, VisualizationModelInterface)


class ArtificialModelInterface(ABC):
    """By inheriting from this interface you indicate that you are providing an artificial parameter dataset,
    and the corresponding artificial data dataset, which can be used to compare the results from epi with the ground truth.
    The comparison can be done using the plotEmceeResults.

    :raises NotImplementedError: Implement the generateArtificialData function to implement this interface.
    """

    NUM_ARTIFICIAL_SAMPLES = 1000

    @abstractmethod
    def generateArtificialData(
        self, numSamples: int = NUM_ARTIFICIAL_SAMPLES
    ) -> None:
        """
        .. note::

            This method returns None. You have to do the following:

            .. code-block:: python

                np.savetxt(trueParams,
                    "Data/" + self.getModelName() + "Params.csv", delimiter=","
                )
                np.savetxt(trueData,
                    "Data/" + self.getModelName() + "Data.csv", delimiter=","
                )

            To create the true data from the true params, you can simply call your model.

        :raises NotImplementedError: Generating the artificial data is up to the user
        """
        raise NotImplementedError

    def paramLoader(self) -> tuple[np.ndarray, np.ndarray]:
        """Load and return all parameters for artificial set ups

        :return: Loaded parameters and the optimal kernel width for each parameter
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        trueParams = np.loadtxt(
            "Data/" + self.getModelName() + "Params.csv", delimiter=","
        )

        if len(trueParams.shape) == 1:
            trueParams = trueParams.reshape((trueParams.shape[0], 1))

        paramStdevs = calcKernelWidth(trueParams)

        return trueParams, paramStdevs


class VisualizationModelInterface(ABC):
    """Provides the function for the generation of the dataGrid and paramGrid for the visualization of the distributions.
    It forces subclasses to implement the abstract methods  getParamBounds: and getDataBounds:.
    """

    @abstractmethod
    def getParamBounds(self) -> np.ndarray:
        """Returns the bounds on the parameters used to visualize the parameter distribution.

        Returns:
              np.array: An array of the form [[lowerLimit_dim1, upperLimit_dim1], [lowerLimit_dim2, upperLimit_dim2],...]

        Raises:
            NotImplementedError: If the method is not implemented in the subclass
        """
        raise NotImplementedError

    @abstractmethod
    def getDataBounds(self) -> np.ndarray:
        """Returns the bounds on the data used to visualize the data distribution.

        Returns:
              np.array: An array of the form [[lowerLimit_dim1, upperLimit_dim1], [lowerLimit_dim2, upperLimit_dim2],...]

        Raises:
            NotImplementedError: If the method is not implemented in the subclass
        """
        raise NotImplementedError

    def scale(interval: np.array, scale: float):
        middle = (interval[1, :] - interval[0, :]) / 2.0
        return (interval - middle) * scale + middle

    def generateVisualizationGrid(
        self, resolution: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """This function creates a grid for the data as well as the parameters with a
        constant number of points in each dimension. It saves the grids as csv files in the `Plots/*grid.csv`
        in your Application folder.

        :param resolution: The number of grid points in each dimension
        :type resolution: int
        :return: The dataGrid and teh paramGrid.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        # allocate storage for the parameter and data plotting grid
        paramGrid = np.zeros((resolution, self.paramDim))
        dataGrid = np.zeros((resolution, self.dataDim))
        for d in range(self.dataDim):
            dataGrid[:, d] = np.linspace(*self.dataBounds[d, :], resolution)
        for d in range(self.paramDim):
            paramGrid[:, d] = np.linspace(*self.paramBounds[d, :], resolution)

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


def autodiff(_cls):
    _cls.initFwAndBw()
    return _cls


class JaxModel(Model):
    """The JaxModel class automatically creates the jacobian method based on the forward method.
    Additionally it jit compiles the forward and jacobian method with jax.
    To use this class you have to implement your forward method using jax, e. g. jax.numpy.

    :param Model: Abstract parent class
    :type Model: _type_
    """

    def __init_subclass__(cls, **kwargs):
        return autodiff(_cls=cls)

    @classmethod
    def initFwAndBw(cls):
        cls.fw = jit(cls.forward)
        cls.bw = jit(jacrev(cls.fw))

    def _forward(self, param):
        return type(self).fw(param)

    def _jacobian(self, param):
        return type(self).bw(param)

    def jacobian(self, param):
        return self._jacobian(param)
