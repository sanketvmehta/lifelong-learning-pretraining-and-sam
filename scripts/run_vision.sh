#!/bin/bash
. activate lll

DATASET=${1}
METHOD=${2}
OUTPUT_DIR=${4}
SEED=${5}
PT=${6}
EPOCHS=${7}
CUDA_DEVICE=${8}
CKPT=${9}
USE_SAM=${10}

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

if [[ $CKPT = 'imagenetinit_ckpt' ]]
then
    CKPT_FLAG="--checkpoint imagenet_pretrain/last_checkpoint.pt --num-excluded-classes 267"
else
    CKPT_FLAG=""
fi

if [[ $USE_SAM = 'sam' ]]
then
    echo ${USE_SAM}
    SAM_FLAG="--sam"
else
    SAM_FLAG=""
fi

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -m img_exps.main_vision \
    --dataset ${DATASET} \
    --method ${METHOD} \
    --data-folder ${DATA_DIR} \
    --output-folder ${OUTPUT_DIR} \
    --seed ${SEED} \
    --batch-size ${BATCH_SIZE} \
    --epochs-per-task ${EPOCHS} \
    --save-models ${PT_FLAG} ${CKPT_FLAG} ${SAM_FLAG} \
    --lr 0.01
