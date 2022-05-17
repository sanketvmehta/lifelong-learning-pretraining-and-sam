import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from img_exps.data.pt_data import get_5_dataset, get_cifar_50, get_split_cifar100
from img_exps.existing_methods.er import ER
from img_exps.existing_methods.ewc import EWC
from img_exps.vision_utils import (
    calculate_run_metrics,
    set_seed,
    extract_logits,
    ResNet,
)
from img_exps.sam import SAM

def disable_running_stats(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.backup_momentum = m.momentum
        m.momentum = 0

def enable_running_stats(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.momentum = m.backup_momentum


def eval_single_epoch(model, loader, criterion, device, task_id=None):
    """
    Evaluate the current model on test dataset of the given task_id

    Args:
        net: Current model
        loader: Test data loader
        criterion: Loss function
        device: device on which to run evaluation
        task_id: Task identity
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X, task_id)
            test_loss += criterion(output, y).item()
            pred = torch.argmax(output, dim=1)
            correct += (pred == y).sum().item()
    test_loss /= len(loader)
    avg_acc = correct / len(loader.dataset)
    return {"accuracy": avg_acc, "loss": test_loss}


def train_single_epoch(
    algo, model, dataloader, criterion, optimizer, classes_per_task, args, task_id=None
):
    """Train model for single epoch.

    Args:
        algo: Algorithm object used during training of model.
        model: Model to be trained.
        dataloader: Train dataloader.
        optimizer: Optimizer to be used in training.
        classes_per_task: How many classes in each task.
        args: Contains other relevant arguments used in training.
        task_id: Which task is currently being trained on.
    """
    model.train()
    for X, y in iter(dataloader):

        if args.method == "sam":
            model.zero_grad()
            X = X.to(args.device)
            y = y.to(args.device)

            # first forward-backward step
            model.apply(disable_running_stats)
            # enable_running_stats(model)  # <- this is the important line
            out = model(X, task_id)
            loss = criterion(out, y)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            model.apply(enable_running_stats)
            # disable_running_stats(model)  # <- this is the important line
            criterion(model(X, task_id), y).mean().backward()
            optimizer.second_step(zero_grad=True)
        else:
            model.zero_grad()
            X = X.to(args.device)
            y = y.to(args.device)
            out = model(X, task_id)
            if args.method == "er":
                if task_id > 0:
                    mem_x, mem_y, mem_task_ids = algo.sample(
                        args.batch_size, exclude_task=None, pr=False
                    )
                    mem_pred = model(mem_x, None)
                    mem_pred = extract_logits(
                        mem_pred, mem_task_ids, classes_per_task, args.device
                    )
                    loss_mem = criterion(mem_pred, mem_y)
                    loss_mem.backward()
                algo.add_reservoir(X, y, None, task_id)
            elif args.method == "ewc":
                loss_ewc = args.lambd * algo.penalty(model)
                loss_ewc.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()


def run_cl(
    args,
    dataloaders,
    task_split,
    num_classes,
    classes_per_task,
    pretrained=False,
    logfile="log.json",
):
    """Runs continual learning.

    Args:
        args: contains all the relevant arguments from command line.
        dataloaders: List of dataloaders that will be used in continual learning.
        num_classes: Total number of classes that will be encountered.
        classes_per_task: Number of classes per task.
        pretrained: Whether to use a pretrained initialization for the ResNet.
        logfile: Where to log results.
    """
    model = ResNet(
        num_classes,
        classes_per_task,
        pretrained=pretrained,
    ).to(device=args.device)
    if args.method == "sam":
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    algo = None
    if args.method == "er":
        algo = ER(args, num_classes)
    elif args.method == "ewc":
        algo = EWC(model, criterion)

    print(task_split)
    full_metrics = {
        "accuracies": defaultdict(list),
        "losses": defaultdict(list),
    }

    for task_id, task in enumerate(task_split):
        print(f"Task {task_id}: {task}")
        lr = max(args.lr * (args.gamma ** task_id), 0.00005)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        train_loader = dataloaders[task_id]["train"]
        iterator = range(args.epochs_per_task)
        if args.epochs_per_task > 1:
            iterator = tqdm(iterator)
        for _ in iterator:
            train_single_epoch(
                algo,
                model,
                train_loader,
                criterion,
                optimizer,
                classes_per_task,
                args,
                task_id,
            )
        if args.method == "ewc":
            loader = torch.utils.data.DataLoader(
                train_loader.dataset, batch_size=200, shuffle=True
            )
            algo.update(model, task_id, loader)

        if args.save_models:
            torch.save(
                {"model": model.state_dict()},
                os.path.join(args.output_folder, "models", f"task_{task_id}_model.pt"),
            )

        # evaluate
        task_average_accuracy = 0
        for eval_task_id in range(task_id + 1):
            test_loader = dataloaders[eval_task_id]["test"]
            metrics = eval_single_epoch(
                model, test_loader, criterion, args.device, eval_task_id
            )
            full_metrics["accuracies"][eval_task_id].append(metrics["accuracy"])
            full_metrics["losses"][eval_task_id].append(metrics["loss"])
            task_average_accuracy += metrics["accuracy"]
        print(
            "TASK {} / {}".format(task_id + 1, len(dataloaders)),
            "\tAvg Acc:",
            task_average_accuracy / (task_id + 1),
        )
    average_accuracy, forgetting, learning_accuracy = calculate_run_metrics(
        full_metrics["accuracies"]
    )
    full_metrics["accuracies"] = dict(full_metrics["accuracies"])
    full_metrics["losses"] = dict(full_metrics["losses"])
    full_metrics["average_accuracy"] = average_accuracy
    full_metrics["forgetting"] = forgetting
    full_metrics["learning_accuracy"] = learning_accuracy
    with open(os.path.join(args.output_folder, logfile), "w") as f:
        json.dump(full_metrics, f, indent=2)
    return full_metrics


def run_lr_hs(
    args,
    dataloaders,
    task_split,
    num_classes,
    classes_per_task,
    pretrained,
):
    """Runs hyperparameter search over learning rates.

    Args:
        args: contains all the relevant arguments from command line.
        dataloaders: List of dataloaders that will be used in hyperparameter search.
        num_classes: Total number of classes that will be encountered.
        task_split: Which task split to use for hyperparameter search.
        classes_per_task: Number of classes per task.
        pretrained: Whether to use a pretrained initialization for the ResNet.
    """
    best_acc = 0
    best_lr = None
    results = []
    for lr in [1e-3, 0.005, 0.01, 0.05, 0.1]:
        args.lr = lr
        print(f"LR: {args.lr}")
        accs = []
        for run in range(args.runs):
            set_seed(args.seed + run)
            metrics = run_cl(
                args,
                dataloaders,
                task_split,
                num_classes,
                classes_per_task,
                pretrained,
                f"log_{run}.json",
            )
            accs.append(metrics["average_accuracy"])
        accuracy = np.mean(accs)
        results.append(metrics)
        if accuracy > best_acc:
            best_acc = accuracy
            best_lr = lr
    print(f"Best LR for {args.method}: {best_lr}")
    with open(os.path.join(args.output_folder, "hs_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="./data")
    parser.add_argument(
        "-d", "--dataset", default="cifar50", choices=["cifar50", "5data", "cifar100"]
    )
    parser.add_argument("--output-folder", default="./out")
    parser.add_argument("-t", "--task-split")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-r", "--runs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--run-hs", action="store_true")
    parser.add_argument("--batch-size", default=10, type=int, help="batch-size")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument(
        "--gamma", default=1.0, type=float, help="lr decay. Use 1.0 for no decay"
    )
    parser.add_argument(
        "--dropout", default=0.0, type=float, help="Use 0.0 for no dropout"
    )
    parser.add_argument(
        "--epochs-per-task", default=1, type=int, help="epochs per task"
    )
    parser.add_argument("--lambd", default=1, type=int, help="EWC")
    parser.add_argument("--mem-size", default=1, type=int, help="mem")
    parser.add_argument("--method", default="sgd", choices=["sgd", "er", "ewc", "sam", "asam", "ssgd"])
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    set_seed(args.seed)

    if args.task_split is not None:
        with open(args.task_split) as f:
            task_split = json.load(f)
    else:
        task_split = None

    # Create dataloaders
    if args.dataset == "cifar50":
        dataloaders, task_split = get_cifar_50(
            args.data_folder,
            args.batch_size,
            args.run_hs,
            saved_tasks=task_split,
        )
        num_classes = 50
        classes_per_task = 10
    elif args.dataset == "5data":
        dataloaders, task_split = get_5_dataset(
            args.data_folder,
            args.batch_size,
            args.run_hs,
            saved_tasks=task_split,
        )
        num_classes = 50
        classes_per_task = 10
    elif args.dataset == "cifar100":
        dataloaders, task_split = get_split_cifar100(
            args.data_folder,
            args.batch_size,
            args.run_hs,
            saved_tasks=task_split,
        )
        num_classes = 100
        classes_per_task = 5
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    print(vars(args))
    os.makedirs(args.output_folder, exist_ok=True)
    with open(os.path.join(args.output_folder, "task_split.json"), "w") as f:
        json.dump(task_split, f, indent=2)
    with open(os.path.join(args.output_folder, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    if args.save_models:
        os.makedirs(os.path.join(args.output_folder, "models"), exist_ok=True)

    if args.dry_run:
        sys.exit()

    if args.run_hs:
        run_lr_hs(
            args,
            dataloaders,
            task_split,
            num_classes,
            classes_per_task,
            args.pretrained,
        )
    else:
        run_cl(
            args,
            dataloaders,
            task_split,
            num_classes,
            classes_per_task,
            args.pretrained,
        )


if __name__ == "__main__":
    main()
