"""
This file creates all the dataloaders used in the vision experiments.
"""
import os
import random
import tarfile

import numpy as np
import torch
import torchvision


class FilteredDataset(torch.utils.data.Dataset):
    """A dataset class used to apply a filter to a dataset based on some metainfo.

    Args:
        dataset: Dataset to be filtered.
        filter_func: Function to apply to metainfo of each example to decide whether
            or not to filter.
        transform: Transform to apply to dataset examples.
        metainfo_func: Takes in the dataset and example index and extracts metainfo.
    """

    def __init__(self, dataset, filter_func=None, transform=None, metainfo_func=None):
        super().__init__()
        self.indices = []
        self.dataset = dataset
        self.filter_func = filter_func
        self.transform = transform
        if metainfo_func is None:
            metainfo_func = lambda dataset, i: dataset[i]
        self.metainfo_func = metainfo_func
        self.filter()

    def filter(self):
        """Iterate through dataset to find which indices to filter"""
        if self.filter_func is None:
            self.indices = list(range(len(self.dataset)))
        else:
            for i in range(len(self.dataset)):
                if self.filter_func(self.metainfo_func(self.dataset, i)):
                    self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x = self.dataset[self.indices[index]]
        if self.transform is not None:
            x = self.transform(x)
        return x


def get_dataloader(dataset, batch_size, shuffle=False):
    """Get dataloader for dataset."""
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4, shuffle=shuffle, pin_memory=True
    )


def create_cifar_filter_func(classes, get_val):
    """Create filter function that takes in the class label and filters if it
    is not one of classes. If get_val is True, subtracts 50 from class label
    first (since validation for CIFAR-50 is done on last 50 classes).
    """
    if get_val:
        return lambda x: (x - 50) in classes
    else:
        return lambda x: x in classes


def create_task_transform(class_to_task_class):
    """Creates a transform that takes in an example and maps the class label according
    to the dictionary class_to_task_class. This is used to change class label to the
    position in the classes list for each task.
    """
    return torchvision.transforms.Lambda(lambda x: (x[0], class_to_task_class[x[1]]))


def get_split_cifar100(dataset_location, batch_size, get_val=False, saved_tasks=None):
    """Get Split CIFAR-100 dataset that is split randomly. Returns the dataloaders
    for each task and the task split.

    Args:
        dataset_location: Where CIFAR-100 dataset is stored or where to download it.
        batch_size: Batch size for all dataloaders.
        get_val: Whether to get validation version of train and test dataloaders.
        saved_tasks: If None, generates a new task split. Otherwise should contain
            the saved task split.
    """
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    test_dataset = torchvision.datasets.CIFAR100(
        dataset_location,
        transform=transform,
        target_transform=None,
        train=False,
    )
    train_dataset = torchvision.datasets.CIFAR100(
        dataset_location,
        transform=transform,
        target_transform=None,
        train=True,
    )

    # Create the task split
    classes = list(range(100))
    random.shuffle(classes)
    tasks = [classes[i * 5 : (i + 1) * 5] for i in range(20)]
    if saved_tasks is not None:
        tasks = saved_tasks
    class_to_task_class = {}
    for task in tasks:
        for task_class, class_id in enumerate(task):
            class_to_task_class[class_id] = task_class

    # Create the dataloaders for each task
    dataloaders = []
    for task_id, task in enumerate(tasks):
        train_subset = FilteredDataset(
            train_dataset,
            create_cifar_filter_func(task, False),
            transform=create_task_transform(class_to_task_class),
            metainfo_func=lambda dataset, i: dataset.targets[i],
        )
        if get_val:
            train_subset, test_subset = get_random_split(train_subset, 0.9)
        else:
            test_subset = FilteredDataset(
                test_dataset,
                create_cifar_filter_func(task, False),
                transform=create_task_transform(class_to_task_class),
                metainfo_func=lambda dataset, i: dataset.targets[i],
            )
        task_dataloaders = {
            "train": get_dataloader(train_subset, batch_size, shuffle=True),
            "test": get_dataloader(test_subset, batch_size, shuffle=False),
        }
        dataloaders.append(task_dataloaders)
    return dataloaders, tasks


def get_cifar_50(dataset_location, batch_size, get_val=False, saved_tasks=None):
    """Get Split CIFAR-50 dataset that is split randomly. Returns the dataloaders
    for each task and the task split.

    Args:
        dataset_location: Where CIFAR-100 dataset is stored or where to download it.
        batch_size: Batch size for all dataloaders.
        get_val: Whether to get validation version of train and test dataloaders.
        saved_tasks: If None, generates a new task split. Otherwise should contain
            the saved task split.
    """
    if get_val:
        target_transform = torchvision.transforms.Lambda(lambda x: x - 50)
    else:
        target_transform = None
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    test_dataset = torchvision.datasets.CIFAR100(
        dataset_location,
        transform=transform,
        target_transform=target_transform,
        train=False,
    )
    train_dataset = torchvision.datasets.CIFAR100(
        dataset_location,
        transform=transform,
        target_transform=target_transform,
        train=True,
    )
    classes = list(range(50))
    random.shuffle(classes)
    tasks = [classes[i * 10 : (i + 1) * 10] for i in range(5)]

    if saved_tasks is not None:
        tasks = saved_tasks
    class_to_task_class = {}
    for task in tasks:
        for task_class, class_id in enumerate(task):
            class_to_task_class[class_id] = task_class

    dataloaders = []
    for task_id, task in enumerate(tasks):
        train_subset = FilteredDataset(
            train_dataset,
            create_cifar_filter_func(task, get_val),
            transform=create_task_transform(class_to_task_class),
            metainfo_func=lambda dataset, i: dataset.targets[i],
        )
        test_subset = FilteredDataset(
            test_dataset,
            create_cifar_filter_func(task, get_val),
            transform=create_task_transform(class_to_task_class),
            metainfo_func=lambda dataset, i: dataset.targets[i],
        )
        task_dataloaders = {
            "train": get_dataloader(train_subset, batch_size, shuffle=True),
            "test": get_dataloader(test_subset, batch_size, shuffle=False),
        }
        dataloaders.append(task_dataloaders)
    return dataloaders, tasks


def get_5_dataset(dataset_location, batch_size, get_val=False, saved_tasks=None):
    """Get 5-dataset that has a random task order. Returns the dataloaders
    for each task and the task order.

    Args:
        dataset_location: Where 5-dataset is stored or where to download it.
        batch_size: Batch size for all dataloaders.
        get_val: Whether to get validation version of train and test dataloaders.
        saved_tasks: If None, generates a new task order. Otherwise should contain
            the saved task order.
    """
    tasks = [
        "cifar10",
        "mnist",
        "svhn",
        "not_mnist",
        "fashion_mnist",
    ]
    random.shuffle(tasks)
    if saved_tasks is not None:
        tasks = saved_tasks
    dataloaders = []
    for task in tasks:
        dataloaders.append(
            get_dataset(task, batch_size, dataset_location, get_val=get_val)
        )
    return dataloaders, tasks


def get_random_split(dataset, split):
    """Randomly splits dataset according to split fraction, and returns the two
    subsets.
    """
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split_index = int(len(indices) * split)
    dataset_1 = torch.utils.data.Subset(dataset, indices[:split_index])
    dataset_2 = torch.utils.data.Subset(dataset, indices[split_index:])
    return dataset_1, dataset_2


def check_not_mnist_files(path):
    """Filters some bad files in NotMNIST dataset."""
    bad_paths = [
        "RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png",
        "Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png",
    ]
    for bad_path in bad_paths:
        if bad_path in path:
            return False
    return True


def get_not_mnist(dataset_location, transform):
    """
    Parses and returns the downloaded notMNIST dataset
    """
    tar_path = os.path.join(dataset_location, "notMNIST_small.tar.gz")
    with tarfile.open(tar_path) as tar:
        tar.extractall(dataset_location)
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(dataset_location, "notMNIST_small"),
        transform=transform,
        is_valid_file=check_not_mnist_files,
    )
    train_dataset, test_dataset = get_random_split(dataset, 0.9)
    return train_dataset, test_dataset


def get_dataset(dataset_name, batch_size, dataset_location, get_val=False):
    """Gets each dataset in 5-dataset.
    Args:
        dataset_name: Which dataset to construct. Must be one of "mnist",
            "fashion_mnist", "not_mnist", "svhn", and "cifar10".
        batch_size: Batch size for all dataloaders.
        dataset_location: Where 5-dataset is stored or where to download it.
        get_val: Whether to get validation version of train and test dataloaders.
    """
    dataset_location = os.path.join(dataset_location, dataset_name)
    transforms_list = [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
    print(dataset_location)
    if dataset_name in ["mnist", "fashion_mnist", "not_mnist"]:
        transforms_list.insert(
            0, torchvision.transforms.Lambda(lambda x: x.convert("RGB"))
        )
    transform = torchvision.transforms.Compose(transforms_list)
    if dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            dataset_location, True, transform=transform, download=False
        )
        test_dataset = torchvision.datasets.CIFAR10(
            dataset_location, False, transform=transform, download=False
        )
    elif dataset_name == "svhn":
        train_dataset = torchvision.datasets.SVHN(
            dataset_location, "train", transform=transform, download=False
        )
        test_dataset = torchvision.datasets.SVHN(
            dataset_location, "test", transform=transform, download=False
        )
    elif dataset_name == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            dataset_location, True, transform=transform, download=False
        )
        test_dataset = torchvision.datasets.MNIST(
            dataset_location, False, transform=transform, download=False
        )
    elif dataset_name == "fashion_mnist":
        train_dataset = torchvision.datasets.FashionMNIST(
            dataset_location, True, transform=transform, download=False
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            dataset_location, False, transform=transform, download=False
        )
    elif dataset_name == "not_mnist":
        train_dataset, test_dataset = get_not_mnist(dataset_location, transform)

    class_to_task_class = list(range(10))
    if get_val:
        train_dataset, test_dataset = get_random_split(train_dataset, 0.9)
    train_dataset = FilteredDataset(
        train_dataset, transform=create_task_transform(class_to_task_class)
    )
    test_dataset = FilteredDataset(
        test_dataset, transform=create_task_transform(class_to_task_class)
    )
    dataloaders = {
        "train": get_dataloader(train_dataset, batch_size, shuffle=True),
        "test": get_dataloader(test_dataset, batch_size),
    }
    return dataloaders
