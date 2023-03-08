import jax.numpy as jnp
import numpy as np

from epi.core.model import ArtificialModelInterface, JaxModel


class Linear(JaxModel, ArtificialModelInterface):
    param_dim = 2
    data_dim = 2

    defaultParamSamplingLimits = np.array([[-10.0, 11.0], [-10.0, 11.0]])
    defaultcentral_param = np.array([0.5, 0.5])

    @classmethod
    def forward(cls, param):
        return jnp.array([param[0] * 10, (-2.0) * param[1] - 2.0])

    def generate_artificial_params(self, num_samples: int):
        return np.random.rand(num_samples, self.param_dim)

    def get_param_bounds(self, scale=1.0) -> np.ndarray:
        return np.array([[-0.2, 1.2], [-0.2, 1.2]])

    def get_data_bounds(self, scale=1.0) -> np.ndarray:
        return np.array([[-2.0, 12.0], [-4.4, -1.6]])


class Exponential(JaxModel):
    param_dim = 2
    data_dim = 2

    defaultParamSamplingLimits = np.array([[1.0, 2.0], [1.0, 2.0]])
    defaultcentral_param = np.array([1.0, 1.0])

    @classmethod
    def forward(cls, param):
        return jnp.array([param[0] * jnp.exp(1), jnp.exp(param[1])])

    def get_param_bounds(self) -> np.ndarray:
        return np.array([0.8, 2.2], [0.8, 2.2])

    # TODO: ???
    # KDExGrid = np.linspace(0.8 * np.exp(1), 2.2 * np.exp(1), KDEresolution)
    # KDEyGrid = np.linspace(np.exp(0.8), np.exp(2.2), KDEresolution)
    # KDExMesh, KDEyMesh = np.meshgrid(KDExGrid, KDEyGrid)
    def get_data_bounds(self) -> np.ndarray:
        return self.forward(self.get_param_bounds())


class LinearODE(JaxModel, ArtificialModelInterface):
    param_dim = 2
    data_dim = 2

    defaultParamSamplingLimits = np.array([[-10.0, 23.0], [-10.0, 23.0]])
    defaultcentral_param = np.array([1.5, 1.5])

    @classmethod
    def forward(cls, param):
        return jnp.array(
            [
                param[0] * jnp.exp(param[1] * 1.0),
                param[0] * jnp.exp(param[1] * 2.0),
            ]
        )

    def generate_artificial_params(self, num_samples: int):
        return np.random.rand(num_samples, 2) + 1
