import jax.numpy as jnp
import numpy as np
import pytest

from eulerpi.data_transformations import DataIdentity
from eulerpi.evaluation.gram_determinant import calc_gram_determinant
from eulerpi.evaluation.kde import GaussKDE
from eulerpi.models import ArtificialModelInterface, JaxModel


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
    from eulerpi.evaluation.transformation import evaluate_density

    param = X2Model.CENTRAL_PARAM
    x2_model = X2Model()
    # Model and calc_gram_determinant has its own tests, so we can use it here to test the transformations
    sim_res, jac = x2_model.forward_and_jacobian(param)
    correction = calc_gram_determinant(jac)

    # KDE has its own tests, so we can use it here to test the transformations
    data = np.array([[0.0], [2.0]])
    data_transformation = DataIdentity()
    kde = GaussKDE(data)
    pure_density = kde(sim_res)

    # Test case 1: When the slice is one dimensional
    slice = np.array([0])
    _, _, density = evaluate_density(
        param, x2_model, data_transformation, kde, slice
    )
    assert density == pure_density * correction

    # Test case 2: When the slice is empty
    slice = np.array([])
    with pytest.raises(IndexError):
        _, _, density = evaluate_density(
            param, x2_model, data_transformation, kde, slice
        )

    # Test case 3: When the slice is two dimensional, but the model is one dimensional
    slice = np.array([0, 1])
    with pytest.raises(IndexError):
        _, _, density = evaluate_density(
            param, x2_model, data_transformation, kde, slice
        )

    # Test case 4: When the param is out of bounds
    slice = np.array([0])
    param = np.array([2.1])
    # Other arguments would change too, but shouldn't matter for this test
    # set logger level to debug to see the warning
    from eulerpi.logger import logger

    logger.setLevel("INFO")
    _, _, density = evaluate_density(
        param, x2_model, data_transformation, kde, slice
    )
    assert density == 0.0
    assert "Parameters outside of predefined range" in caplog.text
