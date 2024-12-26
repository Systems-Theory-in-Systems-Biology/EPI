"""
Test the plotting of samples using the COVID model
"""

import numpy as np

from eulerpi.examples.corona import Corona
from eulerpi.inference import inference
from eulerpi.inferences.inference_type import InferenceType
from eulerpi.plotting import sample_violin_plot


def test_sample_plotting():
    """ """

    np.random.seed(42)
    model = Corona()

    data = np.random.randn(1000, 4) * np.array([1, 5, 25, 2]) + np.array(
        [1, 10, 40, 3]
    )
    print(data.shape)

    num_walkers = 10
    num_steps = 100
    num_burn_in_samples = 10
    thinning_factor = 1

    run_name = "test_run"

    inference(
        model,
        data=data,
        inference_type=InferenceType.SAMPLING,
        slice=np.arange(model.param_dim),
        run_name=run_name,
        num_walkers=num_walkers,
        num_steps=num_steps,
        num_burn_in_samples=num_burn_in_samples,
        thinning_factor=thinning_factor,
    )

    sample_violin_plot(
        model,
        run_name=run_name,
        what_to_plot="param",
        axis_labels=[r"$k_i$", r"$k_d", r"$k_r$"],
    )
    sample_violin_plot(
        model,
        reference_sample=data,
        run_name=run_name,
        what_to_plot="data",
        axis_labels=[r"1", r"2", r"5", r"15 weeks"],
    )
