#!/bin/bash

# 循环从2到50
model=DQNwithHER
# for length in {4..10}; do
#     python main.py minimal_size=128 length=$length use_wandb=false
# done

# for length in {10..20}; do
#     python main.py minimal_size=128 length=$length use_wandb=false
# done

# for length in {20..30}; do
#     python main.py minimal_size=128 length=$length use_wandb=false
# done

# for length in {30..40}; do
#     python main.py minimal_size=128 length=$length use_wandb=false
# done

cd /root/BitFlip/DQNwithHer
CUDA_VISIBLE_DEVICES=1 python main.py minimal_size=64 length=8 use_wandb=true epsilon=0.9 delta_epsilon=1e-5 lr=1e-4