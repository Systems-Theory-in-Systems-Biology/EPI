import numpy as np

from eulerpi.core.data_transformation import DataIdentity
from eulerpi.core.kde import calc_kernel_width
from eulerpi.core.model import Model
from eulerpi.core.transformations import evaluate_density


class CrashModel(Model):
    data_dim = 1
    param_dim = 1

    def forward(self, param):
        raise RuntimeError("Crash in forward eval!")

    def jacobian(self, param):
        return RuntimeError("Crash in jacobian eval!")


def test_logs_error_evaluate_density(caplog):
    """Test that the logger logs an error when the model evaluation fails.

    Args:
        caplog (pytest.LogCaptureFixture): pytest fixture to capture log messages
    """
    central_param = np.array([1.0])
    param_limits = np.array([[0.0, 2.0]])
    crash_model = CrashModel(central_param, param_limits)

    data = np.array([[0.0], [2.0]])
    data_transformation = DataIdentity()
    data_stdevs = calc_kernel_width(data)
    slice = np.array([0])

    density, _ = evaluate_density(
        central_param,
        crash_model,
        data,
        data_transformation,
        data_stdevs,
        slice,
    )
    assert density == 0.0
    assert any(
        [
            record.levelname == "ERROR"
            and "Crash in forward eval!" in record.message
            for record in caplog.records
        ]
    )
