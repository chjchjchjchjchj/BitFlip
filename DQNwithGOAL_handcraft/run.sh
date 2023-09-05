#!/bin/bash

# 循环从2到50
model=DQN
# for length in {2..10}; do
#     # 执行命令，将当前的length值传递给DQN.py
#     python main.py length=$length exp_name="$length bits"
# done

# for length in {10..20}; do
#     # 执行命令，将当前的length值传递给DQN.py
#     python main.py length=$length exp_name="$length bits"
# done

# for length in {20..30}; do
#     # 执行命令，将当前的length值传递给DQN.py
#     python main.py length=$length exp_name="$length bits"
# done

# for length in {30..40}; do
#     # 执行命令，将当前的length值传递给DQN.py
#     python main.py length=$length exp_name="$length bits"
# done
# for length in {4,8,16,32,64}; do
#     for minimal_size in {64,128,256,512}; do
#         python main.py minimal_size=$minimal_size length=$length use_wandb=false
#     done
# done

for length in {4,8,16,32,64}; do
    python main.py minimal_size=128 length=$length use_wandb=false
done
