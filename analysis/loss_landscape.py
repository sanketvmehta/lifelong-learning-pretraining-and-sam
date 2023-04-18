import numpy as np
from analysis.utils import flatten_parameters, assign_params
from tqdm import tqdm
import torch


def calculate_loss_contours(
    model1, model2, model3, dataloader, eval_fn, device, granularity=20, margin=0.2
):
    """Runs the loss contour analysis.
    Creates plane based on the parameters of 3 models, and computes loss and accuracy
    contours on that plane. Specifically, computes 2 axes based on the 3 models, and
    computes metrics on points defined by those axes.
    Args:
        model1: Origin of plane.
        model2: Model used to define y axis of plane.
        model3: Model used to define x axis of plane.
        dataloader: Dataloader for the dataset to evaluate on.
        eval_fn: A function that takes a model, a dataloader, and a device, and returns
            a dictionary with two metrics: "loss" and "accuracy".
        device: Device that the model and data should be moved to for evaluation.
        granularity: How many segments to divide each axis into. The model will be
            evaluated at granularity*granularity points.
        margin: How much margin around models to create evaluation plane.
    """
    w1 = flatten_parameters(model1).to(device=device)
    w2 = flatten_parameters(model2).to(device=device)
    w3 = flatten_parameters(model3).to(device=device)
    model1 = model1.to(device=device)

    # Define x axis
    u = w3 - w1
    dx = torch.norm(u).item()
    u /= dx

    # Define y axis
    v = w2 - w1
    v -= torch.dot(u, v) * u
    dy = torch.norm(v).item()
    v /= dy

    # Define grid representing parameters that will be evaluated.
    coords = np.stack(get_xy(p, w1, u, v) for p in [w1, w2, w3])
    alphas = np.linspace(0.0 - margin, 1.0 + margin, granularity)
    betas = np.linspace(0.0 - margin, 1.0 + margin, granularity)
    losses = np.zeros((granularity, granularity))
    accuracies = np.zeros((granularity, granularity))
    grid = np.zeros((granularity, granularity, 2))

    # Evaluate parameters at every point on grid
    progress = tqdm(total=granularity * granularity)
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            p = w1 + alpha * dx * u + beta * dy * v
            assign_params(model1, p)
            metrics = eval_fn(model1, dataloader, device)
            grid[i, j] = [alpha * dx, beta * dy]
            losses[i, j] = metrics["loss"]
            accuracies[i, j] = metrics["accuracy"]
            progress.update()
    progress.close()
    return {
        "grid": grid.tolist(),
        "coords": coords.tolist(),
        "losses": losses.tolist(),
        "accuracies": accuracies.tolist(),
    }


def get_xy(point, origin, vector_x, vector_y):
    """Return transformed coordinates of a point given parameters defining coordinate
    system.
    Args:
        point: point for which we are calculating coordinates.
        origin: origin of new coordinate system
        vector_x: x axis of new coordinate system
        vector_y: y axis of new coordinate system
    """
    return np.array(
        [
            torch.dot(point - origin, vector_x).item(),
            torch.dot(point - origin, vector_y).item(),
        ]
    )