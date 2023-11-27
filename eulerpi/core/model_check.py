import numpy as np
from jax import vmap

from eulerpi.core.inference import InferenceType, inference
from eulerpi.core.model import JaxModel, Model
from eulerpi.core.plotting import sample_violin_plot


def full_model_check(
    model: Model,
    num_data_points: int = 1000,
    num_model_evaluations: int = 11000,
) -> None:
    """Check your model in a quick run on an artificially created dataset.
    We recommend to run this function for every new model you create.
    It produces a violin plot comparing the artificially created parameters and data to the respectively inferred samples.

    Args:
        model(Model): The model describing the mapping from parameters to data.
        num_data_points (int, optional): The number of data data points to artificially generate (Default value = 1000)
        num_model_evaluations (int, optional): The number of model evaluations to perform in the inference. (Default value = 11000)
    Returns:
        None
    """

    # create artificial parametrs similar to how we create initial walker positions for emcee sampling

    central_param = model.central_param
    param_limits = model.param_limits

    # sample parameters from a uniform distribution around the central parameter and between the parameter limits
    d_min = np.minimum(
        central_param - param_limits[:, 0], param_limits[:, 1] - central_param
    )
    param_sample = central_param + d_min * (
        (np.random.rand(num_data_points, model.param_dim) - 0.5) / 3.0
    )

    # try to use jax vmap to perform the forward pass on multiple parameters at once
    if isinstance(model, JaxModel):
        data_sample = vmap(model.forward, in_axes=0)(param_sample)
    else:
        data_sample = np.vectorize(model.forward, signature="(n)->(m)")(
            param_sample
        )

    # choose sensible values for the sampling hyper-parameters and print them
    num_inference_evaluations = num_model_evaluations - num_data_points

    num_walkers = int(np.sqrt(num_inference_evaluations / 10))
    num_steps = int(num_inference_evaluations / num_walkers)

    num_burn_in_samples = num_walkers
    thinning_factor = int(np.ceil(num_walkers / 10))

    print(f"num_data_points: {num_data_points}")
    print(f"num_walkers: {num_walkers}")
    print(f"num_steps: {num_steps}")
    print(f"num_burn_in_samples: {num_burn_in_samples}")
    print(f"thinning_factor: {thinning_factor}")

    run_name = "test_model_run"

    # perform the inference
    inference(
        model,
        data=data_sample,
        inference_type=InferenceType.MCMC,
        slices=[np.arange(model.param_dim)],
        run_name=run_name,
        num_runs=1,
        num_walkers=num_walkers,
        num_steps=num_steps,
        num_burn_in_samples=num_burn_in_samples,
        thinning_factor=thinning_factor,
    )

    # plot the results
    sample_violin_plot(
        model,
        reference_sample=param_sample,
        run_name=run_name,
        credibility_level=0.999,
        what_to_plot="param",
    )
    sample_violin_plot(
        model,
        reference_sample=data_sample,
        run_name=run_name,
        credibility_level=0.999,
        what_to_plot="data",
    )
