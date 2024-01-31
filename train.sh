#!/bin/bash

ROOT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ -z $1 ]; then
    exit
fi

mkdir -p ${ROOT_DIR}/ckpts

rm -rf ${ROOT_DIR}/ckpts/$1 && MADRONA_MWGPU_KERNEL_CACHE=${ROOT_DIR}/build/cache python ${ROOT_DIR}/scripts/jax_train.py \
    --gpu-sim \
    --ckpt-dir ${ROOT_DIR}/ckpts/$1 \
    --num-worlds 2048 \
    --num-updates 5000 
