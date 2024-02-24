#!/bin/bash

ROOT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ -z $1 ]; then
    exit
fi

if [ -z $2 ]; then
    exit
fi

MADRONA_MWGPU_KERNEL_CACHE=${ROOT_DIR}/build/cache python ${ROOT_DIR}/scripts/jax_infer.py \
    --gpu-sim \
    --ckpt-path ${ROOT_DIR}/ckpts/$2 \
    --num-steps 3600 \
    --crossplay \
    --crossplay-include-past \
    --num-worlds $1 \
    --bf16 \
    --action-dump-path ${ROOT_DIR}/build/actions
    #--print-action-probs \
