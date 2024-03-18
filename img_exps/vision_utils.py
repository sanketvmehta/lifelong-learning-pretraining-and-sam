from functools import partial, partialmethod
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import dropout
import torchvision
from imagenet_pretrain.model import ResNet as PT_ResNet, ResNetBlock, _resnet
from torchvision.models.resnet import ResNet as RNet
from copy import deepcopy


def apply_mask(mem_y, out, n_classes):
    """
    Apply mask on the predicted outputs based on the given task - assuming task-incremental learning setup
    :param mem_y: Actual labels
    :param out: Predictions
    :param n_classes: Number of classes per task
    :return: Masked predictions
    """
    for i, y in enumerate(mem_y):
        mask = torch.zeros_like(out[i])
        mask[y - (y % n_classes) : y - (y % n_classes) + n_classes] = 1
        out[i] = out[i].masked_fill(mask == 0, -1e10)
    return out


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


NUM_EXCLUDED_CLASSES = 267


class ResNet(torch.nn.Module):
    """
    Resnet model used in vision experiments.

    Args:
        total_classes: Total number of classes that will be encountered in all
            tasks in lifelong learning.
        classes_per_task: How many classes are in each task.
        pretrained: Whether to initialize ResNet with pretrained weights.
    """

    def __init__(self, total_classes, classes_per_task, layers=18, pretrained=False,
                 pt_type=None, checkpoint=None, dropout=0.0, num_exclu_classes=None, method=None):
        super(ResNet, self).__init__()

        if num_exclu_classes is None:
            num_exclu_classes = NUM_EXCLUDED_CLASSES
        if checkpoint is not None:
            print("Checkpoint : ", checkpoint)
            print("Total classes : ", 1000 - num_exclu_classes)
            norm_layer = None
            model = PT_ResNet(
                total_classes=1000 - num_exclu_classes,
                layers=layers,
                dropout=dropout,
                norm_layer=norm_layer
            )
            checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
            model_state_dict = checkpoint["model"]
            model.load_state_dict(model_state_dict, strict=False)
            resnet = model.net
        elif pt_type is None:
            if layers == 18:
                norm_layer = None
                resnet = _resnet(
                    "resnet18",
                    ResNetBlock,
                    [2, 2, 2, 2],
                    pretrained,
                    True,
                    dropout=dropout,
                    norm_layer=norm_layer
                )
            elif layers == 34:
                resnet = torchvision.models.resnet34(pretrained=pretrained)
            elif layers == 50:
                resnet = torchvision.models.resnet50(pretrained=pretrained)
            else:
                raise ValueError("not a recognized ResNet")
        elif pt_type == "ssl":
            resnet = torch.hub.load(
                "facebookresearch/semi-supervised-ImageNet1K-models", "resnet18_ssl"
            )
        elif pt_type == "swsl":
            resnet = torch.hub.load(
                "facebookresearch/semi-supervised-ImageNet1K-models", "resnet18_swsl"
            )
        else:
            raise ValueError("not a recognized ResNet")

        for param in resnet.parameters():
            param.requires_grad = True
        self.net = resnet
        self.net.fc = torch.nn.Linear(resnet.fc.in_features, total_classes)
        self.classes_per_task = classes_per_task
        self.task_id = None

    def set_task(self, task_id):
        print("Setting the task_id to :", str(task_id))
        self.task_id = task_id

    def unset_task(self):
        print("Setting the task_id to none")
        self.task_id = None

    def forward(self, x, task_id=None):
        """Runs a forward pass on x with the network. If task_id is None,
        returns all logits. Otherwise, only returns logits corresponding to
        task.
        """
        out = self.net(x)
        if task_id is None and self.task_id is None:
            return out
        elif task_id is None and self.task_id is not None:
            task_id = self.task_id
        start = task_id * self.classes_per_task
        end = (task_id + 1) * self.classes_per_task
        return out[:, start:end]


def set_seed(seed):
    """Set the random seed for the experiment."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
