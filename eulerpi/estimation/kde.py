"""This module provides functions to handle the Kernel Densitiy Estimation (KDE_) in EPI.

    It is used in the EPI algorithm to :py:func:`eulerpi.transformations.evaluate_density <evaluate the density>` of the transformed data distribution at the simulation results.


.. _KDE: https://en.wikipedia.org/wiki/Kernel_density_estimation
"""

from .density_estimator import DensityEstimator
from .kernel_width import calc_silverman_kernel_width


class KDE(DensityEstimator):
    def __init__(self, data, kernel_width_rule=calc_silverman_kernel_width):
        super().__init__()
        self.kernel_width = kernel_width_rule(data)
        self.data = data
