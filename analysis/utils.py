import numpy as np
import torch


def flatten_parameters(model):
    """Returns a flattened tensor containing the parameters of model."""
    return torch.cat([param.flatten() for param in model.parameters()])


def assign_params(model, w):
    """Takes in a flattened parameter vector w and assigns them to the parameters
    of model.
    """
    offset = 0
    for parameter in model.parameters():
        param_size = parameter.nelement()
        parameter.data = w[offset : offset + param_size].reshape(parameter.shape)
        offset += param_size


def flatten_gradients(model):
    """Returns a flattened numpy array with the gradients of the parameters of
    the model.
    """
    return np.concatenate(
        [
            param.grad.detach().cpu().numpy().flatten()
            if param.grad is not None
            else np.zeros(param.nelement())
            for param in model.parameters()
        ]
    )
