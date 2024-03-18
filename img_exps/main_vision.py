import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from img_exps.sam import SAM, disable_running_stats, enable_running_stats

from img_exps.data.pt_data import (
    get_5_dataset,
    get_cifar_50,
    get_split_cifar100,
    get_split_mnist,
)
from img_exps.existing_methods.agem import AGEM
from img_exps.existing_methods.er import ER
from img_exps.existing_methods.ewc import EWC
from img_exps.vision_utils import (
    calculate_run_metrics,
    set_seed,
    extract_logits,
    ResNet,
)


def eval_single_epoch(model, loader, criterion, args, task_id=None):
    """
    Evaluate the current model on test dataset of the given task_id
    :param net: Current model
    :param loader: Test data loader
    :param criterion: Loss function
    :param task_id: Task identity
    :return:
    """
    model.to(args.device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(args.device)
            y = y.to(args.device)
            output = model(X, task_id)
            test_loss += criterion(output, y).item()
            pred = torch.argmax(output, dim=1)
            correct += (pred == y).sum().item()
    test_loss /= len(loader)
    avg_acc = correct / len(loader.dataset)
    return {"accuracy": avg_acc, "loss": test_loss}


def train_single_epoch(
    algo,
    model,
    dataloader,
    criterion,
    optimizer,
    classes_per_task,
    args,
    task_id=None,
    sam=False,
    subset_dataloader=None,
):
    model.train()
    count = 0
    for X, y in iter(dataloader):
        count += 1
        model.zero_grad()
        if sam:
            enable_running_stats(model)

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
        elif args.method == "agem":
            model = algo.observe_agem(model, X, task_id, y)

        if args.method != "agem":
            loss = criterion(out, y)
            loss.backward()
        if sam:
            optimizer.first_step(zero_grad=True)
            disable_running_stats(model)
            criterion(model(X, task_id), y).backward()
            if args.method == "er" and task_id > 0:
                mem_pred = model(mem_x, None)
                mem_pred = extract_logits(
                    mem_pred, mem_task_ids, classes_per_task, args.device
                )
                criterion(mem_pred, mem_y).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()


def run_cl(
    args,
    dataloaders,
    task_split,
    num_classes,
    classes_per_task,
    pretrained=False,
    class_incremental=False,
    sam=False,
    subset_dataloaders=None,
    logfile="log.json",
):
    model = ResNet(
        num_classes,
        classes_per_task,
        layers=args.layers,
        pretrained=pretrained,
        pt_type=args.pt_type,
        checkpoint=args.checkpoint,
        num_exclu_classes=args.num_excluded_classes,
        dropout=args.dropout,
        method=args.method,
    ).to(device=args.device)

    if sam:
        optimizer = SAM(model.parameters(), torch.optim.SGD, rho=args.rho, lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    algo = None
    if args.method == "er":
        algo = ER(args, num_classes)
    elif args.method == "agem":
        algo = AGEM(
            args,
            model,
            optimizer,
            criterion,
            classes_per_task,
            len(dataloaders),
        )
    elif args.method == "ewc":
        algo = EWC(model, criterion)

    print(task_split)
    full_metrics = {
        "accuracies": defaultdict(list),
        "losses": defaultdict(list),
    }

    for task_id, task in enumerate(task_split):
        print(f"Task {task_id}: {task}")

        epochs_per_task = args.epochs_per_task
        lr = max(args.lr * (args.gamma ** task_id), 0.00005)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        train_loader = dataloaders[task_id]["train"]
        subset_dataloader = subset_dataloaders[task_id] if subset_dataloaders is not None else None

        iterator = range(epochs_per_task)
        if epochs_per_task > 1:
            iterator = tqdm(iterator)
        for _ in iterator:
            train_single_epoch(
                algo=algo,
                model=model,
                dataloader=train_loader,
                subset_dataloader=subset_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                classes_per_task=classes_per_task,
                args=args,
                task_id=0 if class_incremental else task_id,
                sam=sam,
            )
        if args.method == "ewc":
            loader = torch.utils.data.DataLoader(
                train_loader.dataset, batch_size=200, shuffle=True
            )
            algo.update(model, 0 if class_incremental else task_id, loader)

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
                model,
                test_loader,
                criterion,
                args,
                0 if class_incremental else eval_task_id,
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
        "-d",
        "--dataset",
        default="cifar50",
        choices=["cifar50", "5data", "cifar100", "mnist"],
    )
    parser.add_argument("--output-folder", default="./out")
    parser.add_argument("--checkpoint")
    parser.add_argument("--num-excluded-classes", type=int, default=267)
    parser.add_argument("-t", "--task-split")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-r", "--runs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--run-hs", action="store_true")
    parser.add_argument("--batch-size", default=10, type=int, help="batch-size")
    parser.add_argument("--layers", default=18, type=int, choices=[18, 34, 50])
    parser.add_argument("--pt-type", choices=["ssl", "swsl"])
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--momentum", default=0.0, type=float, help="momentum")
    parser.add_argument(
        "--gamma", default=1.0, type=float, help="lr decay. Use 1.0 for no decay"
    )
    parser.add_argument(
        "--dropout", default=0.0, type=float, help="Use 0.0 for no dropout"
    )
    parser.add_argument(
        "--epochs-per-task", default=5, type=int, help="epochs per task"
    )
    parser.add_argument("--lambd", default=1, type=int, help="EWC")
    parser.add_argument("--mem-size", default=1, type=int, help="mem")
    parser.add_argument(
        "--method", default="sgd", choices=["sgd", "multi", "er", "ewc", "agem", "ssgd"]
    )
    parser.add_argument("--class-incremental", action="store_true")
    parser.add_argument("--sam", action="store_true")
    parser.add_argument(
        "--rho", default=0.05, type=float, help="neighborhood parameter for sam"
    )
    parser.add_argument("--val", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    set_seed(args.seed)

    if args.task_split is not None:
        with open(args.task_split) as f:
            task_split = json.load(f)
    else:
        task_split = None

    subset_dataloaders = None

    if args.dataset == "cifar50":
        dataloaders, task_split = get_cifar_50(
            args.data_folder,
            args.batch_size,
            args.val,
            saved_tasks=task_split,
        )
        num_classes = 50
        classes_per_task = 10
    elif args.dataset == "5data":
        dataloaders, task_split = get_5_dataset(
            args.data_folder,
            args.batch_size,
            args.val,
            saved_tasks=task_split,
        )
        num_classes = 50
        classes_per_task = 10
    elif args.dataset == "cifar100":
        dataloaders, task_split = get_split_cifar100(
            args.data_folder,
            args.batch_size,
            args.val,
            saved_tasks=task_split,
        )
        num_classes = 100
        classes_per_task = 5
    elif args.dataset == "mnist":
        dataloaders, task_split = get_split_mnist(
            args.data_folder,
            args.batch_size,
            args.val,
            saved_tasks=task_split,
        )
        num_classes = 10
        classes_per_task = 2
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    if args.class_incremental:
        num_classes = classes_per_task
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
            args=args,
            dataloaders=dataloaders,
            task_split=task_split,
            num_classes=num_classes,
            classes_per_task=classes_per_task,
            pretrained=args.pretrained,
            class_incremental=args.class_incremental,
            sam=args.sam,
            subset_dataloaders=subset_dataloaders
        )


if __name__ == "__main__":
    main()