#!/bin/bash
. activate lll

if [[ ${1} = '5data' ]]
then
DATA_DIR=${2}/5data
else
DATA_DIR=${2}/cifar100
fi
RUN_SUBFOLDER=${3}
OUTPUT_FILE=${4}
ANALYSIS=${5}
CUDA_DEVICE=${6}

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -m analysis.run_analysis_vision \
    -d ${DATA_DIR} \
    -r ${RUN_SUBFOLDER} \
    -o ${OUTPUT_FILE} \
    -a ${ANALYSIS} \
    -s 1