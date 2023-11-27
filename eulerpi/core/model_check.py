import jax.numpy as jnp
import numpy as np
from jax import vmap

from eulerpi.core.inference import InferenceType, inference
from eulerpi.core.model import JaxModel, Model
from eulerpi.core.plotting import sample_violin_plot


def basic_model_check(model: Model) -> None:
    """Perform a simple sanity check on the model. It tests the following:
    - The model has a positive parameter dimension
    - The model has a positive data dimension
    - The model has a valid combination of parameter and data dimension
    - The central parameter has the correct shape
    - The parameter limits have the correct shape
    - The model can be instantiated
    - The model forward pass can be calculated
    - The model jacobi matrix can be calculated
    - The return values of the forward pass and the jacobi matrix have the correct shape
    Args:
        model(Model): The model describing the mapping from parameters to data.
    Returns:
        None
    """

    print(
        f"Checking model {model.name} at location \n{model} \nfor basic functionality.\n"
    )

    # test the shapes
    assert (
        model.param_dim > 0
    ), f"Model {model} has a non-positive parameter dimension"
    assert (
        model.data_dim > 0
    ), f"Model {model} has a non-positive data dimension"
    assert model.data_dim >= model.param_dim, (
        f"Model {model} has a data dimension smaller than the parameter dimension. "
        "This is not supported by the inference."
    )
    assert model.central_param.shape == (model.param_dim,), (
        f"Model {model} has a central parameter with the wrong shape. "
        f"Expected {(model.param_dim,)}, got {model.central_param.shape}"
    )
    assert model.param_limits.shape == (model.param_dim, 2), (
        f"Model {model} has parameter limits with the wrong shape. "
        f"Expected {(model.param_dim, 2)}, got {model.param_limits.shape}"
    )

    print("Successfully checked shapes and dimensions of model attributes.\n")
    print(
        f"Evaluate model {model.name} and its jacobian in its central parameter \n{model.central_param}."
    )

    model_forward = model.forward(model.central_param)
    assert (
        model_forward.shape == (1, model.data_dim)
        or model_forward.shape == (model.data_dim,)
        or model_forward.shape == ()
    ), (
        f"Model {model} has a forward function with the wrong shape. "
        f"Expected {(1, model.data_dim)}, {(model.data_dim,)} or {()}, got {model_forward.shape}"
    )

    model_jac = model.jacobian(model.central_param)
    assert (
        model_jac.shape == (model.data_dim, model.param_dim)
        or (model.data_dim == 1 and model_jac.shape == (model.param_dim,))
        or (model.param_dim == 1 and model_jac.shape == (model.data_dim,))
    ), (
        f"Model {model} has a jacobian function with the wrong shape. "
        f"Expected {(model.data_dim, model.param_dim)}, {(model.param_dim,)} or {(model.data_dim,)}, got {model_jac.shape}"
    )

    # check rank of jacobian
    assert jnp.linalg.matrix_rank(model_jac) == model.param_dim, (
        f"The Jacobian of the model {model} does not have full rank. This is a requirement for the inference. "
        "Please check the model implementation."
    )

    fw, jc = model.forward_and_jacobian(model.central_param)
    assert fw.shape == model_forward.shape, (
        f"The shape {fw.shape} of the forward function extracted from the forward_and_jacobian function does not match the shape {model_forward.shape} of the forward function. "
        "Please check the model implementation."
    )
    assert jc.shape == model_jac.shape, (
        f"The shape {jc.shape} of the jacobian extracted from the forward_and_jacobian function does not match the shape {model_jac.shape} of the jacobian. "
        "Please check the model implementation."
    )
    assert jnp.allclose(fw, model_forward), (
        f"The forward function of the model {model} does not match the forward function extracted from the forward_and_jacobian function. "
        "Please check the model implementation."
    )
    assert jnp.allclose(jc, model_jac), (
        f"The jacobian of the model {model} does not match the jacobian extracted from the forward_and_jacobian function. "
        "Please check the model implementation."
    )

    print(
        "Successfully checked model forward simulation and corresponding jacobian.\n"
    )


def inference_model_check(
    model: Model,
    num_data_points: int = 1000,
    num_model_evaluations: int = 11000,
) -> None:
    """Check your model in a quick inference run on an artificially created dataset.
    It produces a violin plot comparing the artificially created parameters and data to the respectively inferred samples.

    Args:
        model(Model): The model describing the mapping from parameters to data.
        num_data_points (int, optional): The number of data data points to artificially generate (Default value = 1000)
        num_model_evaluations (int, optional): The number of model evaluations to perform in the inference. (Default value = 11000)
    Returns:
        None
    """

    print(
        f"Checking model {model.name} at location \n{model} \nfor inference functionality on artificially created data.\n"
    )

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

    print(
        f"Successfully created an artificial data set of size {num_data_points}.\n"
    )

    # choose sensible values for the sampling hyper-parameters and print them
    num_inference_evaluations = num_model_evaluations - num_data_points

    num_walkers = int(np.sqrt(num_inference_evaluations / 10))
    num_steps = int(num_inference_evaluations / num_walkers)

    num_burn_in_samples = num_walkers
    thinning_factor = int(np.ceil(num_walkers / 10))

    print("Attempting inference with hyperparameters chosen as follows:")
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

    print(
        f"Successfully finishes inference run with {num_walkers*num_steps} samples.\n"
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


def full_model_check(
    model: Model,
    num_data_points: int = 1000,
    num_model_evaluations: int = 11000,
) -> None:
    """Check your model for basic functionality and in a quick inference run on an artificially created dataset.
    We recommend to run this function for every new model you create.
    It runs the functions basic_model_check and inference_model_check to perform the checks.

    Args:
        model(Model): The model describing the mapping from parameters to data.
        num_data_points (int, optional): The number of data data points to artificially generate (Default value = 1000)
        num_model_evaluations (int, optional): The number of model evaluations to perform in the inference. (Default value = 11000)
    Returns:
        None
    """

    basic_model_check(model)
    inference_model_check(model, num_data_points, num_model_evaluations)
