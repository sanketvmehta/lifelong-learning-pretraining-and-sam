import copy

import numpy as np
import torch
from scipy import optimize
import logging


def sharpness(model, criterion_fn, A, epsilon=1e-3, p=0, bounds=None):
    """Computes sharpness metric according to https://arxiv.org/abs/1609.04836.

    Args:
        model: Model on which to compute sharpness
        criterion_fn: Function that takes in a model and returns the loss
            value and gradients on the appropriate data that will be used in
            the loss maximization done in the sharpness calculation.
        A: Projection matrix that defines the subspace in which the loss
            maximization will be done. If A=1, no projection will be done.
        epsilon: Defines the size of the neighborhood that will be used in the
            loss maximization.
        p: The dimension of the random projection subspace in which maximization
            will be done. If 0, assumed to be the full parameter space.
    """
    run_fn = create_run_model(model, A, criterion_fn)
    if bounds is None:
        bounds = compute_bounds(model, A, epsilon)
    dim = flatten_parameters(model).shape[0] if p == 0 else p

    # Find the maximum loss in the neighborhood of the minima
    y = optimize.minimize(
        lambda x: run_fn(x),
        np.zeros(dim),
        method="L-BFGS-B",
        bounds=bounds,
        jac=True,
        options={"maxiter": 10},
    ).x.astype(np.float32)

    model_copy = copy.deepcopy(model)
    if A is 1:
        flat_diffs = y
    else:
        flat_diffs = A @ y
    apply_diffs(model_copy, flat_diffs)
    maximum = criterion_fn(model_copy)["loss"]
    loss_value = criterion_fn(model)["loss"]
    sharpness = 100 * (maximum - loss_value) / (1 + loss_value)
    return sharpness


def flatten_parameters(model):
    """Returns a flattened numpy array with the parameters of the model."""
    return np.concatenate(
        [
            param.detach().cpu().numpy().flatten()
            for param in model.parameters()
            if param.requires_grad
        ]
    )


def compute_bounds(model, A, epsilon):
    """Computes the bounds in which to search for the maximum loss."""
    x = flatten_parameters(model)
    if A is 1:
        bounds = epsilon * (np.abs(x) + 1)
    else:
        b, _, _, _ = np.linalg.lstsq(A, x)
        bounds = epsilon * (np.abs(b) + 1)
    return optimize.Bounds(-bounds, bounds)


def create_run_model(model, A, criterion_fn):
    """Creates a run function that takes in parameters in the subspace that loss
    maximization takes place in, and computes the loss and gradients
    corresponding to those parameters.
    """

    def run(y):
        y = y.astype(np.float32)
        model_copy = copy.deepcopy(model)
        model_copy.zero_grad()

        if A is 1:
            flat_diffs = y
        else:
            flat_diffs = A @ y
        apply_diffs(model_copy, flat_diffs)
        metrics = criterion_fn(model_copy)
        objective = -metrics["loss"]
        gradient = -metrics["gradients"]
        logging.info("Loss: %f", objective)
        if A is not 1:
            gradient = gradient @ A
        return objective, gradient.astype(np.float64)

    return run


def apply_diffs(model, diffs):
    """Adds deltas to the parameters in the model corresponding to diffs."""
    parameters = model.parameters()
    idx = 0
    for parameter in parameters:
        if parameter.requires_grad:
            n_elements = parameter.nelement()
            cur_diff = diffs[idx : idx + n_elements]
            parameter.data = parameter.data + torch.tensor(
                cur_diff.reshape(parameter.shape)
            ).to(device=parameter.device)
            idx += n_elements
