import argparse
import torchvision
import os
import requests


def download_data(dataset, folder):
    if dataset == "5data":
        torchvision.datasets.CIFAR10(
            os.path.join(folder, "cifar10"), True, download=True
        )
        torchvision.datasets.CIFAR10(
            os.path.join(folder, "cifar10"), False, download=True
        )
        torchvision.datasets.SVHN(os.path.join(folder, "svhn"), "train", download=True)
        torchvision.datasets.SVHN(os.path.join(folder, "svhn"), "test", download=True)
        torchvision.datasets.MNIST(os.path.join(folder, "mnist"), True, download=True)
        torchvision.datasets.MNIST(os.path.join(folder, "mnist"), False, download=True)
        torchvision.datasets.FashionMNIST(
            os.path.join(folder, "fashion_mnist"), True, download=True
        )
        torchvision.datasets.FashionMNIST(
            os.path.join(folder, "fashion_mnist"), False, download=True
        )
        url = "http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz"
        r = requests.get(url, allow_redirects=True)
        open(os.path.join(folder, "not_mnist", "notMNIST_small.tar.gz"), "wb").write(
            r.content
        )
    else:
        torchvision.datasets.CIFAR100(folder, True, download=True)
        torchvision.datasets.CIFAR100(folder, False, download=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["5data", "cifar100"], required=True)
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    download_data(args.dataset, args.folder)


if __name__ == "__main__":
    main()
