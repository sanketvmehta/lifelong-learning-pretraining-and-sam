import argparse
import json
import logging
import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from img_exps.data import pt_data
from img_exps.vision_utils import ResNet
from tqdm import tqdm

from analysis.sharpness import sharpness
from analysis.utils import flatten_gradients

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def load_checkpoint(model, checkpoint):
    """Loads checkpoint into model"""
    state = torch.load(checkpoint)
    model.load_state_dict(state["model"])


def create_eval_fn(task_id, calculate_gradient=False):
    """Creates an evaluation function for a given task. Returns an evaluation
    function that takes in a model, dataloader, and device, and evaluates the
    model on the data from the dataloader. Returns a dictionary with mean
    "loss" and "accuracy". If calculate_gradient is True, dictionary will also
    contain gradients for the model wrt the loss on the data.


    Args:
        task_id: Task id corresponding to the data that will be evaluated.
        calculate_gradient: Whether gradient should be calculated.
    """

    def eval_fn(model, dataloader, device):
        model.eval()
        total_loss = 0
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum").to(device=device)
        num_correct = 0
        model.zero_grad()
        torch.set_grad_enabled(calculate_gradient)
        for X, y in iter(dataloader):
            X = X.to(device=device)
            y = y.to(device=device)
            output = model(X, task_id)
            preds = torch.argmax(output, dim=1)
            num_correct += (preds == y).sum().item()
            loss = loss_fn(output, y) / len(dataloader.dataset)
            if calculate_gradient:
                loss.backward()
            total_loss += loss.item()
        accuracy = num_correct / len(dataloader.dataset)
        metrics = {"loss": total_loss, "accuracy": accuracy}
        if calculate_gradient:
            gradients = flatten_gradients(model)
            metrics["gradients"] = gradients

        return metrics

    return eval_fn


def create_sharpness_fn(dataloader, task_id, device):
    """Creates an evaluation function for the sharpness calculation.
    Essentially modifies the signature for the eval_fn returned by create_eval_fn
    by only taking in a model.
    """
    full_loss_fn = create_eval_fn(task_id, calculate_gradient=True)

    def get_loss(model):
        return full_loss_fn(model, dataloader, device)

    return get_loss


def run_sharpness(model, model_checkpoints, dataloaders, device, p=100):
    """Calculates the sharpness of each model checkpoint on the corresponding
    dataset. Only run it for the first 5 models.

    Args:
        model: Instantiation of the model to calculate sharpness for.
        model_checkpoints: Checkpoint files that will be loaded into model before
            calculating sharpness
        dataloaders: Dataloaders corresponding to each checkpoint in
            model_checkpoints
        device: Device that all evaluation should be done on.
        p: Projection dimension that will be used when sharpness is calculated.
            If p=0, no projecting will be done.
    """
    model_list = {i: model_checkpoints[i] for i in range(5)}
    num_parameters = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    if p == 0:
        A = 1
    else:
        # Randomly sample projection matrix
        A = np.random.rand(num_parameters, p)
        A = A.astype(np.float32)
        A /= np.linalg.norm(A, axis=0, keepdims=True)
    progress = tqdm(total=2 * len(model_list))
    results = defaultdict(dict)
    for model_idx in model_list:
        task_idx = model_idx
        load_checkpoint(model, model_checkpoints[model_idx])
        for epsilon in [1e-3, 5e-4]:
            sharpness_value = sharpness(
                model,
                create_sharpness_fn(dataloaders[task_idx]["test"], task_idx, device),
                A,
                epsilon=epsilon,
                p=p,
            )
            logging.info(
                f"Model {model_idx} Epsilon {epsilon} sharpness: {sharpness_value}"
            )
            results[model_idx][epsilon] = sharpness_value
            progress.update()
    return dict(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-folder", required=True)
    parser.add_argument("-d", "--data-folder", required=True)
    parser.add_argument("-o", "--output-file", default="./out.json")
    parser.add_argument("-s", "--start", type=int, default=0)
    parser.add_argument("-e", "--end", type=int, default=1)
    parser.add_argument(
        "-a", "--analysis", default="lmi", choices=["lmi", "contour", "sharpness"]
    )

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(args.run_folder, "task_split.json")) as f:
        task_split = json.load(f)
    with open(os.path.join(args.run_folder, "config.json")) as f:
        config = json.load(f)
    dataset = config["dataset"]

    # Get dataloaders
    if dataset == "5data":
        dataloaders, _ = pt_data.get_5_dataset(
            args.data_folder, 500, saved_tasks=task_split
        )
        total_num_classes = 50
        classes_per_task = 10
    elif dataset == "cifar50":
        dataloaders, _ = pt_data.get_cifar_50(
            args.data_folder, 500, saved_tasks=task_split
        )
        total_num_classes = 50
        classes_per_task = 10
    elif dataset == "cifar100":
        dataloaders, _ = pt_data.get_split_cifar100(
            args.data_folder, 500, saved_tasks=task_split
        )
        total_num_classes = 100
        classes_per_task = 5
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    # Get checkpoints
    model_checkpoints = [
        os.path.join(args.run_folder, "models", f"task_{m}_model.pt")
        for m in range(len(os.listdir(os.path.join(args.run_folder, "models"))))
    ]
    model = ResNet(total_num_classes, classes_per_task).to(device=args.device)

    # Run analysis
    if args.analysis == "sharpness":
        results = run_sharpness(model, model_checkpoints, dataloaders, args.device)
    else:
        raise ValueError(f"Analysis type {args.analysis} not supported")

    with open(args.output_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
