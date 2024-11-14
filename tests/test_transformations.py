import jax.numpy as jnp
import numpy as np
import pytest

from eulerpi.core.data_transformations import DataIdentity
from eulerpi.core.kde import calc_kernel_width, eval_kde_gauss
from eulerpi.core.models import ArtificialModelInterface, JaxModel
from eulerpi.core.transformations import calc_gram_determinant


def test_calc_gram_determinant():
    # Test case 1: When the jacobian is a square matrix
    jac = jnp.array([[1, 2], [3, 4]])
    expected_result = jnp.abs(jnp.linalg.det(jac))
    assert expected_result == 2.0
    assert calc_gram_determinant(jac) == expected_result

    # Test case 2: When the jacobian is not a square matrix, and the columns are linearly independent
    jac = jnp.array([[1, 0, 0], [0, 1, 0]]).T
    expected_result = jnp.sqrt(jnp.linalg.det(jnp.matmul(jac.T, jac)))
    assert expected_result == 1.0
    assert calc_gram_determinant(jac) == expected_result

    # Test case 3: When the jacobian is a zero matrix
    jac = jnp.zeros((2, 2))
    expected_result = 0.0
    assert calc_gram_determinant(jac) == expected_result

    # Test case 4: When the jacobian has negative determinant
    jac = jnp.array([[2, 1], [1, 2]])
    expected_result = jnp.abs(jnp.linalg.det(jac))
    assert expected_result == 3.0
    assert calc_gram_determinant(jac) == expected_result


class X2Model(JaxModel, ArtificialModelInterface):
    param_dim = 1
    data_dim = 1
    CENTRAL_PARAM = np.array([1.0])
    PARAM_LIMITS = np.array([[0.0, 2.0]])

    def __init__(self):
        super(JaxModel, self).__init__(self.CENTRAL_PARAM, self.PARAM_LIMITS)

    @classmethod
    def forward(cls, param):
        return param**2

    def generate_artificial_params(self, num_samples: int) -> jnp.ndarray:
        return np.random.randn(num_samples, self.param_dim)


def test_evaluate_density(caplog):
    from eulerpi.core.transformations import evaluate_density

    param = X2Model.CENTRAL_PARAM
    x2_model = X2Model()
    # Model and calc_gram_determinant has its own tests, so we can use it here to test the transformations
    sim_res, jac = x2_model.forward_and_jacobian(param)
    correction = calc_gram_determinant(jac)

    # KDE has its own tests, so we can use it here to test the transformations
    data = np.array([[0.0], [2.0]])
    data_transformation = DataIdentity()
    data_stdevs = calc_kernel_width(data)
    pure_density = eval_kde_gauss(data, sim_res, data_stdevs)

    # Test case 1: When the slice is one dimensional
    slice = np.array([0])
    density, _ = evaluate_density(
        param, x2_model, data, data_transformation, data_stdevs, slice
    )
    assert density == pure_density * correction

    # Test case 2: When the slice is empty
    slice = np.array([])
    with pytest.raises(IndexError):
        density, _ = evaluate_density(
            param, x2_model, data, data_transformation, data_stdevs, slice
        )

    # Test case 3: When the slice is two dimensional, but the model is one dimensional
    slice = np.array([0, 1])
    with pytest.raises(IndexError):
        density, _ = evaluate_density(
            param, x2_model, data, data_transformation, data_stdevs, slice
        )

    # Test case 4: When the param is out of bounds
    slice = np.array([0])
    param = np.array([2.1])
    # Other arguments would change too, but shouldn't matter for this test
    # set logger level to debug to see the warning
    from eulerpi import logger

    logger.setLevel("INFO")
    density, _ = evaluate_density(
        param, x2_model, data, data_transformation, data_stdevs, slice
    )
    assert density == 0.0
    assert "Parameters outside of predefined range" in caplog.text
