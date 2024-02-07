#!/bin/bash

ROOT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ -z $1 ]; then
    exit
fi

mkdir -p ${ROOT_DIR}/ckpts

rm -rf ${ROOT_DIR}/ckpts/$1 && MADRONA_MWGPU_KERNEL_CACHE=${ROOT_DIR}/build/cache python ${ROOT_DIR}/scripts/jax_train.py \
    --gpu-sim \
    --ckpt-dir ${ROOT_DIR}/ckpts/$1 \
    --num-updates 50000 \
    --num-worlds 2048 \
    --lr 5e-5 \
    --steps-per-update 40 \
    --num-bptt-chunks 4 \
    --num-minibatches 4 \
    --entropy-loss-coef 0.01 \
    --value-loss-coef 0.5 \
    --num-channels 512 \
    --pbt-ensemble-size 4 \
    --pbt-past-policies 32 \
    --bf16
