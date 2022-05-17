import torch
import torchvision
import numpy as np
import random


def extract_logits(out, task_ids, n_classes, device):
    """
    Extract logits corresponding to task_ids from out.

    Args:
        out: Predictions
        task_ids: Task ids
        n_classes: Number of classes per task
    """
    indices = (
        (torch.arange(n_classes * out.size(0)) % n_classes)
        .reshape(out.size(0), n_classes)
        .to(device=device)
    )
    indices = indices + (task_ids * n_classes).unsqueeze(1)
    return out[torch.arange(out.size(0)).unsqueeze(1), indices]


def calculate_run_metrics(accuracies):
    """Calculates average accuracy, forgetting, and learning accuracy given accuracies on each task
    for model after every task.
    """
    average_accuracy = 0
    forgetting = 0
    learning_accuracy = 0
    for task_id in accuracies:
        average_accuracy += accuracies[task_id][-1]
        forgetting += max(accuracies[task_id]) - accuracies[task_id][-1]
        learning_accuracy += accuracies[task_id][0]
    return (
        average_accuracy / len(accuracies),
        forgetting / len(accuracies),
        learning_accuracy / len(accuracies),
    )


class ResNet(torch.nn.Module):
    """
    Resnet model used in vision experiments.

    Args:
        total_classes: Total number of classes that will be encountered in all
            tasks in lifelong learning.
        classes_per_task: How many classes are in each task.
        pretrained: Whether to initialize ResNet with pretrained weights.
    """

    def __init__(self, total_classes, classes_per_task, pretrained=False):
        super(ResNet, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)

        for param in resnet.parameters():
            param.requires_grad = True
        self.net = resnet
        self.net.fc = torch.nn.Linear(resnet.fc.in_features, total_classes)
        self.classes_per_task = classes_per_task

    def forward(self, x, task_id=None):
        """Runs a forward pass on x with the network. If task_id is None,
        returns all logits. Otherwise, only returns logits corresponding to
        task.
        """
        out = self.net(x)
        if task_id is None:
            return out
        start = task_id * self.classes_per_task
        end = (task_id + 1) * self.classes_per_task
        return out[:, start:end]


def set_seed(seed):
    """Set the random seed for the experiment."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
