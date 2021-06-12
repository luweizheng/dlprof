#!/bin/bash

#SBATCH --job-name=image_nnbench
#SBATCH --nodes=1
#SBATCH --partition=tesla
#SBATCH --gpus=1

# set up environment
source activate torch16


amp_level="O2"
arch="resnet56"
platform="gpu"
intput_size=32
cd ..

for batch_size in 512 # 256 512 64 128 256 512
do
    filename=bs_${batch_size}-dlprof
    folder=output/${platform}/cifar10_${intput_size}_train_1p/${filename}
    echo ${filename}
    mkdir -p ${folder}
    dlprof -f true --mode pytorch --reports=summary,detail,iteration,kernel,tensor \ 
        --delay 60 --duration 60 python3 -u train_1p_cifar10.py \
        --platform ${platform} \
        --device-id 0 \
        --arch ${arch} \
        --input_size ${intput_size} \
        --workers 32 \
        --data "~/Datasets/CIFAR10/" \
        --save-dir ${folder} \
        --batch-size ${batch_size} \
        --epochs 5 \
        --print-freq 1000 \
        --amp  \
        --amp-level ${amp_level} \
        > ${folder}/train.log 2>&1 &
    wait
done