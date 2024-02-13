#!/bin/bash

ROOT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ -z $1 ]; then
    exit
fi

MADRONA_MWGPU_KERNEL_CACHE=${ROOT_DIR}/build/cache python ${ROOT_DIR}/scripts/jax_infer.py \
    --gpu-sim \
    --ckpt-path ${ROOT_DIR}/ckpts/$1 \
    --num-steps 1000 \
    --single-policy 2 \
    --num-worlds 1 \
    --bf16 \
    --print-action-probs \
    --action-dump-path ${ROOT_DIR}/build/actions
