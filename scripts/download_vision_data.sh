#!/bin/bash
. activate lll

DATASET=${1}

if [[ $DATASET = '5data' ]]
then
    mkdir -p data/5data/cifar10
    mkdir -p data/5data/svhn
    mkdir -p data/5data/mnist
    mkdir -p data/5data/fashion_mnist
    mkdir -p data/5data/not_mnist
    python -m img_exps.data.download_data --data 5data --folder data/5data
else
    mkdir -p data/cifar100
    python -m img_exps.data.download_data --data cifar100 --folder data/cifar100
fi