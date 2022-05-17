#!/bin/bash
. activate lll

DATASET=${1}
METHOD=${2}
OUTPUT_DIR=${4}
SEED=${5}
PT=${6}
EPOCHS=${7}
CUDA_DEVICE=${8}

if [[ $DATASET = '5data' ]]
then
    BATCH_SIZE=64
    DATA_DIR="${3}/5data"
else
    BATCH_SIZE=10
    DATA_DIR="${3}/cifar100"
fi

if [[ $PT = 'pt' ]]
then
    PT_FLAG="--pretrained"
else
    PT_FLAG=""
fi

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -m img_exps.main_vision \
    --dataset ${DATASET} \
    --method ${METHOD} \
    --data-folder ${DATA_DIR} \
    --output-folder ${OUTPUT_DIR} \
    --seed ${SEED} \
    --batch-size ${BATCH_SIZE} \
    --epochs-per-task ${EPOCHS} \
    --save-models ${PT_FLAG} \
    --lr 0.01
